import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter
from scipy.fftpack import ifft, fft, ifft2, fftshift
from scipy import interpolate
from scipy.interpolate import griddata, interp1d
import time
import h5py
import os.path as path
from pymongo import MongoClient
import json
from ast import literal_eval
import unicodedata
from bson import json_util
from datetime import datetime
import os

np.seterr(divide = 'ignore', invalid = 'ignore')
date_format = "%Y-%m-%d"
time_format = "%H:%M:%S"
c0  =  299792458.0

class jro_gbsar_processor():
	def __init__(self, db_name, collection_name, algorithm):
		self.db_name = db_name
		self.collection_name = collection_name
		self.algorithm = algorithm
		self.db_client = MongoClient() #connect to MongoClient

	def read_data(self):
		db = self.db_client[self.db_name] #read 'sar_database'
		self.sar_collection = db[self.collection_name] #read collection, ex: '1493324602284'
		experiment_data = self.sar_collection.find()
		self.ntakes = len(self.sar_collection.find().distinct('take_number'))

		print "Found %d takes to process." %(self.ntakes)

		parameters = experiment_data[0]

		self.beam_angle = parameters['beam_angle']
		self.xai = float(parameters['start_position'])
		self.xaf = float(parameters['stop_position'])
		self.dax = float(parameters['delta_position'])
		self.nx = int(parameters['nposition'])
		self.fre_min = float(parameters['start_freq'])
		self.fre_max = float(parameters['stop_freq'])
		self.nfre = int(parameters['nfreq'])
		self.fre_c = (self.fre_min+self.fre_max) / 2.0
		self.df = (self.fre_max - self.fre_min) / (self.nfre - 1.0)

	def process_data(self, xi, xf, yi, yf, dx, dy, R0 = 0.0, ifft_fact = 8, win = False):
		#grid extension: [(xi, xf), (yi, yf)]
		#grid resolution: dx and dy
		self.xi = xi
		self.xf = xf
		self.yi = yi
		self.yf = yf
		self.dx = dx
		self.dy = dy

		results_folder = "/home/andre/sar_processed_data/imaging/" + self.collection_name

		if not os.path.exists(results_folder):
			os.makedirs(results_folder)

		processed_data_db = self.db_client['sar_processed_data']
		sar_processed_data_collection = processed_data_db[self.collection_name]

		if self.algorithm == "range_migration":
			range_res = 2 ** 11
			cross_range_res = 2 ** 13
			rows = int((max(cross_range_res, self.nx) - self.nx) / 2)
			cols = int((max(range_res, self.nfre) - self.nfre) / 2)
			cols_left = cols
			cols_right = cols
			rows_up = rows
			rows_down = rows

			if not (self.nx % 2) == 0:
				rows_down = rows + 1

			if not (self.nfre % 2) == 0:
				cols_left = cols+1

			rs = (self.yf + self.yi) / 2.0
			#rs = 100.0

			s0 = self.calculate_matched_filter(rs, range_res, cross_range_res)

			for take in range(1, self.ntakes + 1):
				print "Processing %d out of %d." %(take, self.ntakes)
				s21 = np.empty([self.nx, self.nfre], dtype = np.complex64)
				data = self.sar_collection.find({'take_number' : str(take + 1)})

				for position in range(self.nx):
					print "Take %d, position %d" %(take, position)
					data_real = literal_eval(data[position]['data']['data_real'])
					data_imag = literal_eval(data[position]['data']['data_imag'])
					s21[position,:] = np.array(data_real)[0] + 1j * np.array(data_imag)[0]

				if win:
					s21 = self.hanning(s21)

				s21 = np.pad(s21,[[rows_up,rows_down],[cols_left,cols_right]], 'constant', constant_values=0)

				dt = json.loads(data[self.nx-1]['datetime'], object_hook=json_util.object_hook)
				date = str(dt.date())
				time = str(dt.time().strftime("%H:%M:%S"))

				self.rm_algorithm(s21, s0, take, date, time)

		if self.algorithm == "terrain_mapping":
			self.nposx =  int(np.ceil((xf - xi) / dx) + 1) #number of positions axis x
			self.nposy =  int(np.ceil((yf - yi) / dy) + 1) #number of positions axis y
			self.xf    =  self.xi + self.dx * (self.nposx - 1) #recalculating x final position
			self.yf    =  self.yi + self.dy * (self.nposy - 1) #recalculating y final position
			self.npos  =  self.nposx * self.nposy #total of positions
			self.nr    =  2 ** int(np.ceil(np.log2(self.nfre*ifft_fact))) #calculate a number of ranges,
																#considering the zero padding
			n  = np.arange(self.nr) #final number of ranges
			B  = self.df * self.nr #final bandwidth
			dr = c0 / (2*B) #recalculate resolution, considering ifft_fact
			self.rn = dr * n #range vector
			self.R  = dr * self.nr #for the period verb for the interp

			xa = self.xai + self.dax * np.arange(self.nx) #antenna positions vector
			xn = self.xi  + self.dx  * np.arange(self.nposx) #grid vector axis x
			yn = self.yi  + self.dy  * np.arange(self.nposy) #grid vector axis y

			sar_processed_data_collection.delete_many({})

			Rnk = self.calculate_Rnk(xn, yn ,xa) #vector of distance from the antenna positions to the grid
			Rnk_folder = results_folder + "/Rnk.hdf5"
			f = h5py.File(Rnk_folder, 'w')
			dset = f.create_dataset("Rnk", (Rnk.shape), dtype = np.float32)
			dset[...] = Rnk
			f.close()
			post = {"type": "parameters",
					"xi": self.xi,
					"xf": self.xf,
					"yi": self.yi,
					"yf": self.yf,
					"dx": self.dx,
					"dy": self.dy,
					"fi": self.fre_min,
					"ff": self.fre_max,
					"nfre": self.nfre,
					"ifft_fact": ifft_fact,
					"window": win,
					"Rnk_folder": Rnk_folder}

			sar_processed_data_collection.insert(post)

			parameters = sar_processed_data_collection.find_one({'type' : 'parameters'})
			file_temp = h5py.File(parameters['Rnk_folder'], 'r')
			dset = file_temp["Rnk"]
			Rnk = dset[...]
			file_temp.close()

			starting_take = 1

			'''
			if sar_processed_data_collection.find().count() is not 0:
				parameters = sar_processed_data_collection.find_one({'type' : 'parameters'})

				if (self.xi == parameters['xi']) and (self.xf == parameters['xf']) and (self.yi == parameters['yi']) and (self.yf == parameters['yf']):
					file_temp = h5py.File(parameters['Rnk_folder'], 'r')
					dset = file_temp["Rnk"]
					Rnk = dset[...]
					file_temp.close()
					starting_take = len(self.sar_collection.find().distinct('take'))
				else:
					sar_processed_data_collection.delete_many({})

			if sar_processed_data_collection.find().count() is 0:
				Rnk = self.calculate_Rnk(xn, yn ,xa) #vector of distance from the antenna positions to the grid
				Rnk_folder = results_folder + "/Rnk.hdf5"
				f = h5py.File(Rnk_folder, 'w')
				dset = f.create_dataset("Rnk", (Rnk.shape), dtype = np.float32)
				dset[...] = Rnk
				f.close()
				post = {"type": "parameters",
						"xi": self.xi,
	                    "xf": self.xf,
	            	    "yi": self.yi,
						"yf": self.yf,
						"dx": self.dx,
						"dy": self.dy,
						"ifft_fact": ifft_fact,
						"window": win,
						"Rnk_folder": Rnk_folder}

				sar_processed_data_collection.insert(post)
				post = None
			'''
			for take in range(starting_take, self.ntakes + 1):
				print "Processing %d out of %d." %(take, self.ntakes)
				s21 = np.empty([self.nx, self.nfre], dtype = np.complex64)
				data = self.sar_collection.find({'take_number' : str(take)})

				#if data.count() < self.nx:
				#	continue

				for position in range(self.nx):
					data_real = literal_eval(data[position]['data']['data_real'])
					data_imag = literal_eval(data[position]['data']['data_imag'])
					s21[position,:] = np.array(data_real)[0] + 1j * np.array(data_imag)[0]

				if win:
					s21 = s21 * np.hanning(s21.shape[1])
					s21 = s21 * np.hanning(s21.shape[0])[:,np.newaxis]

				dt = datetime.strptime(data[self.nx-1]['datetime'], "%Y-%m-%d %H:%M:%S.%f")
				date = str(dt.date())
				time = str(dt.time().strftime("%H:%M:%S"))

				self.tm_algorithm(s21, Rnk, take, date, time, results_folder, sar_processed_data_collection)

	def calculate_matched_filter(self, rs, range_res, cross_range_res):
		k = (4 * np.pi / c0) * np.linspace((self.fre_min), (self.fre_max), range_res)
		ku = (np.pi / self.dax) * np.linspace(-1, 1, cross_range_res) * (1/2**3)
		kk, kuu = np.meshgrid(k, ku)
		self.kxmn = np.sqrt(kk ** 2 - kuu ** 2)
		self.kxmn = np.nan_to_num(self.kxmn)
		self.kymin = np.amin(ku)
		self.kymax = np.amax(ku)
		self.kxmin = np.amin(self.kxmn)
		self.kxmax = np.amax(self.kxmn)
		print self.kxmin, self.kxmax

		s0 = np.exp(1j * rs * (self.kxmn - kk))
		return s0

	def calculate_Rnk(self, xn, yn, xa):
		Rnk = np.zeros([self.nx, self.npos], dtype = np.float32)
		for k in range(self.nx):
			for y in range(self.nposy):
				Rnk[k, y * self.nposx: (y+1) * self.nposx] = np.sqrt((xn - xa[k])**2 + yn[y]**2)
		return Rnk

	def hanning(self, s21):
		Wr = 1.0 - np.cos(2 * np.pi * np.arange(1,self.nfre+1)/(self.nfre+1)) #Window Range
		s21 = s21 * Wr
		Wx = 1.0 - np.cos(2 * np.pi * np.arange(1,self.nx+1)/(self.nx+1)) #Window Cross-range
		s21 = s21 * Wx[:, np.newaxis]
		return s21

	def tm_algorithm(self, s21, Rnk, take, date, time, results_folder, results_collection):
		I = np.zeros([self.npos], dtype = np.complex64)
		s21_arr = np.zeros([self.nx, self.nr], dtype = np.complex64)
		nc0 = int(self.nfre/2.0) #first chunk of the frequency: f0,fc
		nc1 = int((self.nfre+1)/2.0) #first chunk of the frequency: fc,ff
		s21_arr[:,0:nc1] = s21[:, nc0:self.nfre] #invert array order
		s21_arr[:,self.nr - nc0: self.nr] = s21[:, 0:nc0]
		Fn0 = self.nr * ifft(s21_arr, n = self.nr)

		for k in range(0,self.nx):
			Fn = np.interp(Rnk[k,:] - R0, self.rn, np.real(Fn0[k,:])) + 1j * np.interp(Rnk[k,:] - R0, self.rn, np.imag(Fn0[k,:]))
			Fn *= np.exp(4j * np.pi * (self.fre_min/c0) * (Rnk[k,:] - R0))
			I +=  Fn

		I /= (self.nfre * self.nx)
		I = np.reshape(I, (self.nposy, self.nposx))
		I = np.flipud(I)

		data_results_folder = results_folder + "/data"
		images_results_folder = results_folder + "/images"

		if not os.path.exists(data_results_folder):
			os.makedirs(data_results_folder)

		if not os.path.exists(images_results_folder):
			os.makedirs(images_results_folder)

		f = h5py.File(data_results_folder + "/image%d.hdf5" %take, 'w')
		dset = f.create_dataset("Complex_image", (self.nposy, self.nposx), dtype = np.complex64)
		dset.attrs['dx'] = self.dx
		dset.attrs['dy'] = self.dy
		dset.attrs['date'] = date
		dset.attrs['time'] = time
		dset.attrs['ntoma'] = take
		dset.attrs['xi'] = self.xi
		dset.attrs['xf'] = self.xf
		dset.attrs['yi'] = self.yi
		dset.attrs['yf'] = self.yf
		dset.attrs['fi'] = self.fre_min
		dset.attrs['ff'] = self.fre_max
		dset.attrs['beam_angle'] = self.beam_angle
		dset[...] = I
		f.close()

		post = {"type": "data",
				"take": str(take),
				"date": date,
				"time": time,
				"route": data_results_folder + "/image%d.hdf5" %take}

		results_collection.insert(post)
		post = None

		I = 10 * np.log10(np.absolute(I))

		fig = plt.figure(1)

		if take == 1:
			self.vmin = np.amin(I) + 28
			self.vmax = np.amax(I)

			aux = int((self.yf - (self.xf * np.tan(48.0 * np.pi /180.0))) * I.shape[0] / (self.yf - self.yi))
			mask = np.zeros(I.shape)

			count = 0

			for k in range(self.nposy):
				#if k >= (int(0.5 * self.nposx / np.tan(32.0 * np.pi /180.0)) + 1):
				if k >= (aux + 1):
					mask[k, 0:count] = 1
					mask[k, self.nposx - count -1:self.nposx-1] = 1
					count = count + 1
			self.masked_values = np.ma.masked_where(mask == 0, mask)

		im = plt.imshow(I, cmap = 'jet', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf], vmin = self.vmin, vmax = self.vmax)
		plt.imshow(self.masked_values, cmap = 'Greys', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf], vmin = self.vmin, vmax = self.vmax, interpolation = 'none')
		cbar = plt.colorbar(im, orientation = 'vertical')
		plt.ylabel('Range (m)', fontsize = 14)
		plt.xlabel('Cross-range (m)', fontsize = 14)
		plt.savefig(images_results_folder + '/image%d.png' %take)
		fig.clear()

	def rm_algorithm(self, s21, s0, take, date, time):
		Fmn = np.fft.fftshift(fft(s21, axis = 0)) * s0
		ky_len = len(s0)

		res = 2 ** 9
		kx_even = np.linspace(self.kxmin, self.kxmax, res)
		Fst = np.zeros((ky_len, res), dtype = np.complex64)

		for i in range(ky_len):
			interp_fn = interp1d(self.kxmn[i], Fmn[i], bounds_error = False, fill_value=(0,0))
			Fst[i] = interp_fn(kx_even)

		Fst = np.nan_to_num(Fst)

		f = ifft2(Fst)
		f = (np.rot90(f))
		#f = np.flipud(f)

		max_range = 2 * np.pi * f.shape[0]/(self.kxmax - self.kxmin)
		max_crange = 2 * np.pi * f.shape[1]/(self.kymax - self.kymin)
		print max_range, max_crange

		I = 10 * np.log10(np.absolute(f))
		#I = (np.absolute(f))

		fig = plt.figure(1)

		#if take == 0:
		#	return

		if take == 0:
			self.vmin = np.amin(I)
			self.vmax = np.amax(I)

		im = plt.imshow(I, cmap = 'jet', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf], vmin = self.vmin, vmax = self.vmax)
		cbar = plt.colorbar(im, orientation = 'vertical')
		plt.ylabel('Range (m)', fontsize = 14)
		plt.xlabel('Cross-range (m)', fontsize = 14)
		plt.show()
		fig.clear()

class jro_sliding_processor():
	def __init__(self, db_name, collection_name):
		self.db_name = db_name
		self.collection_name = collection_name
		self.db_client = MongoClient() #connect to MongoClient

		self.results_folder = "/home/andre/sar_processed_data/sliding/" + self.collection_name

		if not os.path.exists(self.results_folder):
			os.makedirs(self.results_folder)

	def read_data(self):
		db = self.db_client[self.db_name] #read 'sar_processed_data'
		self.sar_collection = db[self.collection_name]
		parameters = self.sar_collection.find({'type' : 'parameters'})[0]
		self.ntakes = len(self.sar_collection.find().distinct('take'))

		print "Found %d takes to process." %(self.ntakes - 1)

		self.xi = float(parameters['xi'])
		self.xf = float(parameters['xf'])
		self.yi = float(parameters['yi'])
		self.yf = float(parameters['yf'])
		self.fre_min = 15500000000.0
		#self.fre_min = float(parameters['fi'])
		self.fre_max = 15600000000.0
		#self.fre_max = float(parameters['ff'])
		self.fre_c = (self.fre_min+self.fre_max) / 2.0
		self.lambda_d = 1000 * (c0/(self.fre_c * 4 * np.pi))

	def calculate_sliding(self, threshold, output_images):
		self.threshold = threshold

		print "Processing data..."

		correlation = self.calculate_correlation()

		aux = int((self.yf - (self.xf * np.tan(48.0 * np.pi /180.0))) * correlation.shape[0] / (self.yf - self.yi))

		mask_aux = np.zeros(correlation.shape)

		count = 0
		nposx = correlation.shape[1]
		nposy = correlation.shape[0]

		for k in range(nposy):
			if k >= (aux + 1):
				mask_aux[k, 0:count] = 1
				mask_aux[k, nposx - count -1:nposx-1] = 1
				count = count + 1
		masked_values = np.ma.masked_where(mask_aux == 0, mask_aux)

		fig = plt.figure(1)
		plt.title("Complex correlation magnitude", fontsize = 11)
		im = plt.imshow(np.absolute(correlation), cmap = 'jet', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf] , vmin = 0, vmax = 1)
		plt.imshow(masked_values, cmap = 'Greys', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])
		plt.colorbar(im, orientation = 'vertical', format='%.1f')
		plt.savefig(self.results_folder + "/complex_correlation_mag.png")
		fig.clear()

		mask = np.absolute(correlation)
		mask[mask < threshold] = 0
		mask[mask >= threshold] = 1

		vmax = np.pi * self.lambda_d
		vmin = -1 * vmax

		mag_mean = np.zeros(self.ntakes - 1)
		phase_mean = np.zeros(self.ntakes - 1)
		phase_mean_sum = np.zeros(self.ntakes - 1)
		phase_std_dev = np.zeros(self.ntakes - 1)
		date_values = []

		fig = plt.figure(1)

		file_temp = h5py.File(self.results_folder + "/mask.hdf5", 'w')
		dset = file_temp.create_dataset("Mask", (mask.shape), dtype = np.uint8)
		dset[...] = mask
		file_temp.close()

		for take in range(1, self.ntakes):
			print "Processing take %d and %d." %(take, take + 1)
			data = self.sar_collection.find_one({'take' : str(take)})
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_master = dset[...]
			file_temp.close()

			data = self.sar_collection.find_one({'take' : str(take + 1)})
			date = (data['date'])
			time = (data['time'])
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_slave = dset[...]
			file_temp.close()

			phase = np.angle(Imagen_master * np.conj(Imagen_slave))
			magnitude = np.absolute(Imagen_master)
			masked_angle = np.ma.masked_where(mask == 0, phase)
			masked_magnitude = np.ma.masked_where(mask == 0, magnitude)

			mag_mean[take - 1] = masked_magnitude.mean()
			phase_mean[take - 1] = masked_angle.mean() * self.lambda_d
			phase_mean_sum[take - 1] = np.sum(phase_mean)
			phase_std_dev[take - 1] = np.std(masked_angle) * self.lambda_d
			date_values.append(datetime.strptime(''.join((date, time)), ''.join((date_format, time_format))))

			if output_images:
				fig.suptitle("Image %d and %d" %(take, take + 1))
				plt.ylabel('Range (m)', fontsize = 14)
				plt.xlabel('Cross-range (m)', fontsize = 14)

				if take == 1:
					im = plt.imshow(self.lambda_d * phase, cmap = 'jet', aspect = 'auto',
									extent = [self.xi,self.xf,self.yi,self.yf], vmin = vmin, vmax = vmax)
					cbar = plt.colorbar(im, orientation = 'vertical', format='%.2f')
				im = plt.imshow(self.lambda_d * phase, cmap = 'jet', aspect = 'auto',
								extent = [self.xi,self.xf,self.yi,self.yf], vmin = vmin, vmax = vmax)
				plt.savefig(self.results_folder + "/take%d_%d.png" %(take, (take+1)))

		fig.clear()
		fig = plt.figure(figsize = (10.0, 6.0))

		plt.subplot(221)
		plt.title(r'$\overline{\Delta r}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\overline{\Delta r}\/\/(mm)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_mean)
		ax = plt.gca()
		ax.set_ylim([-(np.amax(phase_mean) * 2.0), (np.amax(phase_mean) * 2.0)])
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		plt.subplot(222)
		plt.title(r'$\overline{\Delta r_{acc}}\/\/(mm)\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\overline{\Delta r_{acc}}$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_mean_sum)
		ax = plt.gca()
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		sub = plt.subplot(223)
		plt.title(r'$\sigma_{\Delta r}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\sigma_{\Delta r}\/\/(mm)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_std_dev)
		ax = plt.gca()
		ax.set_ylim([0.0 , np.amax(phase_std_dev) * 1.2])
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		plt.subplot(224)
		plt.title(r'$\overline{mag}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\overline{mag}$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, mag_mean)
		ax = plt.gca()
		ax.set_ylim([0.0 , np.amax(mag_mean) * 1.2])
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels,  rotation=30, fontsize=10)

		plt.tight_layout()
		plt.savefig(self.results_folder + "/statistical_report.png")

		print "Done!"

	def calculate_correlation(self):
		for take in range(1, self.ntakes):
			data = self.sar_collection.find_one({'take' : str(take)})
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_master = dset[...]
			file_temp.close()

			data = self.sar_collection.find_one({'take' : str(take + 1)})
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_slave = dset[...]
			file_temp.close()

			if take == 1:
				num = Imagen_master * np.conj(Imagen_slave)
				den = np.sqrt(np.absolute(Imagen_master)**2 * np.absolute(Imagen_slave)**2)
				continue
			num += Imagen_master * np.conj(Imagen_slave)
			den += np.sqrt(np.absolute(Imagen_master)**2 * np.absolute(Imagen_slave)**2)

		return num/den

	def clean_data_report(self, threshold):
		self.clean_data_list = []
		file_temp = h5py.File(self.results_folder + "/mask.hdf5", 'r')
		dset = file_temp["Mask"]
		mask = dset[...]
		file_temp.close()
		clean_data_set = set()

		for take in range(1, self.ntakes):
			data = self.sar_collection.find_one({'take' : str(take)})
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_master = dset[...]
			file_temp.close()

			data = self.sar_collection.find_one({'take' : str(take + 1)})
			date = (data['date'])
			time = (data['time'])
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_slave = dset[...]
			file_temp.close()

			phase = np.angle(Imagen_master * np.conj(Imagen_slave))
			masked_angle = np.ma.masked_where(mask == 0, phase)
			phase_std_dev = np.std(masked_angle) * self.lambda_d

			if phase_std_dev < threshold:
				#clean_data_list.append([take, take +1])
				clean_data_set.add(take)
				clean_data_set.add(take + 1)

		self.clean_data_list = list(clean_data_set)

		clean_results_folder = self.results_folder + "/clean_data"
		if not os.path.exists(clean_results_folder):
			os.makedirs(clean_results_folder)

		report_len = len(self.clean_data_list) - 1
		mag_mean = np.zeros(report_len)
		phase_mean = np.zeros(report_len)
		phase_mean_sum = np.zeros(report_len)
		phase_std_dev = np.zeros(report_len)
		date_values = []

		fig = plt.figure(1)

		vmax = np.pi * self.lambda_d
		vmin = -1 * vmax

		for index, element in enumerate(self.clean_data_list):
			if index == len(self.clean_data_list) - 1:
				break
			master = self.clean_data_list[index]
			slave = self.clean_data_list[index + 1]

			print "Processing take %d and %d." %(master, slave)

			data = self.sar_collection.find_one({'take' : str(master)})
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_master = dset[...]
			file_temp.close()

			data = self.sar_collection.find_one({'take' : str(slave)})
			date = (data['date'])
			time = (data['time'])
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_slave = dset[...]
			file_temp.close()

			phase = np.angle(Imagen_master * np.conj(Imagen_slave))
			magnitude = np.absolute(Imagen_master)
			masked_angle = np.ma.masked_where(mask == 0, phase)
			masked_magnitude = np.ma.masked_where(mask == 0, magnitude)

			mag_mean[index] = masked_magnitude.mean()
			phase_mean[index] = masked_angle.mean() * self.lambda_d
			phase_mean_sum[index] = np.sum(phase_mean)
			phase_std_dev[index] = np.std(masked_angle) * self.lambda_d
			date_values.append(datetime.strptime(''.join((date, time)), ''.join((date_format, time_format))))

		fig.clear()
		fig = plt.figure(figsize = (15.0, 8.0))

		plt.subplot(221)
		plt.title(r'$\overline{\Delta r}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\overline{\Delta r}\/\/(mm)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_mean)
		ax = plt.gca()
		ax.set_ylim([-(np.amax(phase_mean) * 2.0), (np.amax(phase_mean) * 2.0)])
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		plt.subplot(222)
		plt.title(r'$\overline{\Delta r}\/\/(acc)\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\overline{\Delta r}\/\/(acc)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_mean_sum)
		ax = plt.gca()
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		sub = plt.subplot(223)
		plt.title(r'$\sigma_{\Delta r}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\sigma_{\Delta r}\/\/(mm)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_std_dev)
		ax = plt.gca()
		ax.set_ylim([0.0 , np.amax(phase_std_dev) * 1.2])
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		plt.subplot(224)
		plt.title(r'$\overline{mag}\/\/vs\/\/time\/\/(normalized)$', fontsize = 16)
		plt.ylabel(r'$\overline{mag}$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, mag_mean)
		ax = plt.gca()
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels,  rotation=30, fontsize=10)

		plt.tight_layout()
		plt.savefig(clean_results_folder + "/statistical_report.png")

		print "Done!"

	def calculate_integrated_sliding(self, integration_factor):
		file_temp = h5py.File(self.results_folder + "/mask.hdf5", 'r')
		dset = file_temp["Mask"]
		mask = dset[...]
		file_temp.close()

		n = len(self.clean_data_list) - (len(self.clean_data_list) % integration_factor)
		self.clean_data_list = self.clean_data_list[:n]
		groups_list = [self.clean_data_list[i: i + integration_factor] for i in range(0, len(self.clean_data_list), integration_factor)]

		integrated_sliding_results_folder = self.results_folder + "/integrated_sliding_factor_%d" %(integration_factor)
		if not os.path.exists(integrated_sliding_results_folder):
			os.makedirs(integrated_sliding_results_folder)

		integrated_sliding_len = len(groups_list) - 1
		mag_mean = np.zeros(integrated_sliding_len)
		phase_mean = np.zeros(integrated_sliding_len)
		phase_mean_sum = np.zeros(integrated_sliding_len)
		phase_std_dev = np.zeros(integrated_sliding_len)
		date_values = []

		fig = plt.figure(1)

		vmax = np.pi * self.lambda_d
		vmin = -1 * vmax

		for enum, group in enumerate(groups_list):
			if enum == len(groups_list) - 1:
				break
			for index, element in enumerate(groups_list[enum]):
				data = self.sar_collection.find_one({'take' : str(element)})
				file_temp = h5py.File(data['route'], 'r')
				dset = file_temp["Complex_image"]
				if index == 0:
					Imagen_master = dset[...]
					file_temp.close()
					continue
				Imagen_master += dset[...]
				file_temp.close()
			Imagen_master /= integration_factor

			for index, element in enumerate(groups_list[enum + 1]):
				data = self.sar_collection.find_one({'take' : str(element)})
				file_temp = h5py.File(data['route'], 'r')
				dset = file_temp["Complex_image"]
				if index == 0:
					Imagen_slave = dset[...]
					file_temp.close()
					continue
				Imagen_slave += dset[...]
				file_temp.close()
				if index == len(group) - 1:
					date = (data['date'])
					time = (data['time'])
			Imagen_slave /= integration_factor

			phase = np.angle(Imagen_master * np.conj(Imagen_slave))
			magnitude = np.absolute(Imagen_master)
			masked_angle = np.ma.masked_where(mask == 0, phase)
			masked_magnitude = np.ma.masked_where(mask == 0, magnitude)

			mag_mean[enum] = masked_magnitude.mean()
			phase_mean[enum] = masked_angle.mean() * self.lambda_d
			phase_mean_sum[enum] = np.sum(phase_mean)
			phase_std_dev[enum] = np.std(masked_angle) * self.lambda_d
			date_values.append(datetime.strptime(''.join((date, time)), ''.join((date_format, time_format))))

			#fig.suptitle("Image %d and %d" %(take, take + 1))
			plt.ylabel('Range (m)', fontsize = 14)
			plt.xlabel('Cross-range (m)', fontsize = 14)

			if enum == 0:
				im = plt.imshow(self.lambda_d * phase, cmap = 'jet', aspect = 'auto',
								extent = [self.xi,self.xf,self.yi,self.yf], vmin = vmin, vmax = vmax)
				cbar = plt.colorbar(im, orientation = 'vertical', format='%.2f')
			im = plt.imshow(self.lambda_d * phase, cmap = 'jet', aspect = 'auto',
							extent = [self.xi,self.xf,self.yi,self.yf], vmin = vmin, vmax = vmax)
			plt.savefig(integrated_sliding_results_folder + "/groups_%d_%d_takes%s_%s.png" %(enum, enum+1, tuple(groups_list[enum]), tuple(groups_list[enum + 1])))

		fig.clear()
		fig = plt.figure(figsize = (15.0, 8.0))

		plt.subplot(221)
		plt.title(r'$\overline{\Delta r}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\overline{\Delta r}\/\/(mm)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_mean)
		ax = plt.gca()
		ax.set_ylim([-(np.amax(phase_mean) * 2.0), (np.amax(phase_mean) * 2.0)])
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		plt.subplot(222)
		plt.title(r'$\overline{\Delta r}\/\/(acc)\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\overline{\Delta r}\/\/(acc)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_mean_sum)
		ax = plt.gca()
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		sub = plt.subplot(223)
		plt.title(r'$\sigma_{\Delta r}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\sigma_{\Delta r}\/\/(mm)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_std_dev)
		ax = plt.gca()
		ax.set_ylim([0.0 , np.amax(phase_std_dev) * 1.2])
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		plt.subplot(224)
		plt.title(r'$\overline{mag}\/\/vs\/\/time\/\/(normalized)$', fontsize = 16)
		plt.ylabel(r'$\overline{mag}$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, mag_mean * 2)
		ax = plt.gca()
		ax.set_ylim([0.0 , np.amax(mag_mean * 2) * 1.2])
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		plt.tight_layout()
		plt.savefig(integrated_sliding_results_folder + "/statistical_report.png")
		print "Done!"

	def calculate_integrated_sliding2(self, integration_factor, output_images):
		file_temp = h5py.File(self.results_folder + "/mask.hdf5", 'r')
		dset = file_temp["Mask"]
		mask = dset[...]
		file_temp.close()

		new_clean_data_list = []

		for index, element in enumerate(self.clean_data_list):
			if index == len(self.clean_data_list) - 1:
				break
			new_clean_data_list.append([self.clean_data_list[index], self.clean_data_list[index + 1]])

		n = len(new_clean_data_list) - (len(new_clean_data_list) % integration_factor)
		new_clean_data_list = new_clean_data_list[:n]
		groups_list = [new_clean_data_list[i: i + integration_factor] for i in range(0, len(new_clean_data_list), integration_factor)]

		integrated_sliding2_results_folder = self.results_folder + "/integrated_sliding2_factor_%d" %(integration_factor)
		if not os.path.exists(integrated_sliding2_results_folder):
			os.makedirs(integrated_sliding2_results_folder)

		integrated_sliding2_len = len(groups_list) - 1
		mag_mean = np.zeros(integrated_sliding2_len)
		phase_mean = np.zeros(integrated_sliding2_len)
		phase_mean_sum = np.zeros(integrated_sliding2_len)
		phase_std_dev = np.zeros(integrated_sliding2_len)
		date_values = []

		vmax = np.pi * self.lambda_d
		vmin = -1 * vmax

		for enum, group in enumerate(groups_list):
			if enum == len(groups_list) - 1:
				break

			for index, element in enumerate(groups_list[enum]):
				data0 = self.sar_collection.find_one({'take' : str(element[0])})
				file_temp0 = h5py.File(data0['route'], 'r')
				dset0 = file_temp0["Complex_image"]
				data1 = self.sar_collection.find_one({'take' : str(element[1])})
				file_temp1 = h5py.File(data1['route'], 'r')
				dset1 = file_temp1["Complex_image"]

				if index == 0:
					Imagen = np.sqrt(dset0[...] * np.conj(dset1[...]))
					date_begin = data0['date']
					time_begin = data0['time']
				else:
					Imagen += np.sqrt(dset0[...] * np.conj(dset1[...]))
				file_temp0.close()
				file_temp1.close()
				if index == len(group) - 1:
					date = data1['date']
					time = data1['time']
			Imagen /= integration_factor

			phase = np.angle(Imagen)
			magnitude = np.absolute(Imagen)
			masked_angle = np.ma.masked_where(mask == 0, phase)
			masked_magnitude = np.ma.masked_where(mask == 0, magnitude)
			masked_plot = np.ma.masked_where(mask == 1, magnitude)

			mag_mean[enum] = masked_magnitude.mean()
			phase_mean[enum] = masked_angle.mean() * self.lambda_d
			phase_mean_sum[enum] = np.sum(phase_mean)
			phase_std_dev[enum] = np.std(masked_angle) * self.lambda_d
			date_values.append(datetime.strptime(''.join((date_begin, time_begin)), ''.join((date_format, time_format))))

			if output_images:
				fig = plt.figure(1)
				plt.title("Integration factor: {integration_factor} \n From {date_begin} {time_begin} \n to {date} {time} (UTC)".format(integration_factor = integration_factor,
						  date_begin = date_begin, time_begin = time_begin, date = date, time= time), fontsize = 11)
				plt.ylabel('Range (m)', fontsize = 12)
				plt.xlabel('Cross-range (m)', fontsize = 12)

				if enum == 0:
					im = plt.imshow(self.lambda_d * phase, cmap = 'jet', aspect = 'auto',
									extent = [self.xi,self.xf,self.yi,self.yf], vmin = vmin, vmax = vmax)
					cbar = plt.colorbar(im, orientation = 'vertical', format='%.2f')
					cbar.ax.set_title('Displacement \n (mm)', fontsize = 10)

					aux = int((self.yf - (self.xf * np.tan(48.0 * np.pi /180.0))) * Imagen.shape[0] / (self.yf - self.yi))
					mask_aux = np.zeros(Imagen.shape)

					count = 0
					nposx = Imagen.shape[1]
					nposy = Imagen.shape[0]

					for k in range(nposy):
						if k >= (aux + 1):
							mask_aux[k, 0:count] = 1
							mask_aux[k, nposx - count -1:nposx-1] = 1
							count = count + 1
					masked_values = np.ma.masked_where(mask_aux == 0, mask_aux)
					plt.imshow(masked_values, cmap = 'binary', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])
					plt.imshow(masked_plot, cmap = 'binary', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])

				if enum > 0:
					im = plt.imshow(self.lambda_d * phase, cmap = 'jet', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf], vmin = vmin, vmax = vmax)
					plt.imshow(masked_values, cmap = 'binary', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])
					plt.imshow(masked_plot, cmap = 'binary', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])

				plt.savefig(integrated_sliding2_results_folder + "/groups_%d_%d.png" %(enum, enum+1))

		fig = plt.figure(1)
		fig.clear()
		fig = plt.figure(figsize = (10.0, 6.0))

		plt.subplot(221)
		plt.title(r'$\overline{\Delta r}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\overline{\Delta r}\/\/(mm)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_mean)
		ax = plt.gca()
		ax.set_ylim([-(np.amax(phase_mean) * 2.0), (np.amax(phase_mean) * 2.0)])
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		plt.subplot(222)
		plt.title(r'$\overline{\Delta r_{acc}}\/\/(mm)\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\overline{\Delta r}\/\/(acc)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_mean_sum)
		ax = plt.gca()
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		sub = plt.subplot(223)
		plt.title(r'$\sigma_{\Delta r}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\sigma_{\Delta r}\/\/(mm)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		plt.plot(date_values, phase_std_dev)
		ax = plt.gca()
		ax.set_ylim([0.0 , np.amax(phase_std_dev) * 1.2])
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		plt.subplot(224)
		plt.title(r'$\overline{mag}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\overline{mag}$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		#plt.plot(date_values, mag_mean * 2 * 1e6)
		plt.plot(date_values, mag_mean)
		ax = plt.gca()
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		#ax.set_ylim([0.0 , np.amax(mag_mean * 2 * 1e6) * 1.2])
		ax.set_ylim([0.0 , np.amax(mag_mean) * 1.2])
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		plt.tight_layout()
		plt.savefig(integrated_sliding2_results_folder + "/statistical_report.png")
		print "Done!"

if __name__ == "__main__":
	xi =  -250.0
	xf =  250.0
	yi =  50.0
	yf =  550.0

	R0 	=  0.0
	dx  =  1.0
	dy  =  1.0

	collection_name = 'jro-labtest-2017-07-2722:40:48.212442'
	db_name = 'sar_processed_data'
	db_name = 'sar_database'
	algorithm = 'terrain_mapping'

	dp = jro_gbsar_processor(db_name = db_name, collection_name = collection_name, algorithm = algorithm)
	dp.read_data()
	dp.process_data(xi = xi, xf = xf, yi = yi, yf = yf, dx = dx, dy = dy, ifft_fact = 8, win = True)
	'''

	collection_name = 'jro-labtest-2017-07-2722:40:48.212442'
	db_name = 'sar_processed_data'

	x = jro_sliding_processor(db_name = db_name, collection_name = collection_name)
	x.read_data()
	x.calculate_sliding(0.85, output_images = False)
	x.clean_data_report(0.7)
	x.calculate_integrated_sliding2(integration_factor = 4, output_images = False)
	#x.calculate_integrated_sliding2(integration_factor = 1)
	'''
