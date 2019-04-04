import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

from matplotlib.dates import DayLocator, HourLocator, DateFormatter
from scipy.fftpack import ifft, fft, ifft2, fftshift
from scipy import interpolate
from scipy.interpolate import griddata, interp1d

import ast, time, h5py, json, unicodedata, os

from pymongo import MongoClient
from ast import literal_eval
from bson import json_util
from datetime import datetime, timedelta

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(threshold=np.nan)
date_format="%Y-%m-%d"
time_format="%H:%M:%S"
c0=299792458.0
RAW_DATA_PATH='/home/andre/sar_raw_data'
IMAGING_RESULTS_PATH='/home/andre/sar_processed_data/imaging'
SLIDING_RESULTS_PATH='/home/andre/sar_processed_data/sliding'

class jro_gbsar_processor():
	#def __init__(self, db_name, collection_name, algorithm):
	def __init__(self, db_name=None, collection_name=None, single_file=None, raw_data_path=None, results_path=None):
		if db_name is not None:
			self.db_name=db_name
			self.db_client=MongoClient() #connect to MongoClient
			db=self.db_client[self.db_name]
		if collection_name is not None:
			self.collection_name=collection_name
			self.sar_collection = db[self.collection_name]
		if single_file is not None:
			self.single_file=single_file
		if raw_data_path==None:
			self.raw_data_path=RAW_DATA_PATH
		if results_path==None:
			self.results_path=IMAGING_RESULTS_PATH

	def insert_data_db(self):
		print("Only not existing files will be inserted...")
		for (dirpath, dirnames, filenames) in os.walk(os.path.join(self.raw_data_path, self.collection_name)):
			for element in filenames:
				file_location=os.path.join(dirpath, element)
				tmp=h5py.File(file_location, 'r+')
				dset=tmp['sar_dataset']

				#insert metadata to the collection (if does not exist)
				if not self.sar_collection.find_one({'_id':'config'}):
					post={'_id':'config',
						  'start_position':dset.attrs['xi'],
						  'stop_position':dset.attrs['xf'],
						  'npos':int(dset.attrs['npos']),
						  'delta':dset.attrs['dx'],
						  'start_freq':float(dset.attrs['fi']),
						  'stop_freq':float(dset.attrs['ff']),
						  'nfre':int(dset.attrs['nfre']),
						  'beam_angle':int(dset.attrs['beam_angle'])}
					self.sar_collection.insert_one(post)
		        #insert data to the collection
				if not self.sar_collection.find_one({'path':file_location}):
					print "Inserting {} to {} db!".format(file_location, self.db_name)
					post={'type':'data',
						  'path':file_location,
						  'datetime':dset.attrs['datetime'],
						  'take_index':int(dset.attrs['take_index'])}
					self.sar_collection.insert_one(post)
				tmp.close()
		print "Done!"

	def read_data(self):
		parameters=self.sar_collection.find_one({"_id":"config"})
		self.ntakes_list=sorted(self.sar_collection.find({"type":"data"}).distinct('take_index'), key=int)
		self.ntakes=len(self.ntakes_list)

		print "Found %d takes to process." %(self.ntakes)

		self.beam_angle=parameters['beam_angle']
		self.xai=float(parameters['start_position'])
		self.xaf=float(parameters['stop_position'])
		self.dax=float(parameters['delta'])
		self.nx=parameters['npos']
		self.fre_min=parameters['start_freq']
		self.fre_max=parameters['stop_freq']
		self.nfre=parameters['nfre']
		self.fre_c=(self.fre_min+self.fre_max) / 2.0
		self.df=(self.fre_max - self.fre_min) / (self.nfre - 1.0)

	def plot_data_profiles(self):
		results_folder=os.path.join("/home/andre/sar_processed_data/data_profiles/", self.collection_name)

		if not os.path.exists(results_folder):
			os.makedirs(results_folder)

		starting_take=1

		for take in range(starting_take, self.ntakes + 1):
			print "Processing %d out of %d." %(take, self.ntakes)
			#s21 = np.empty([self.nx, self.nfre], dtype = np.complex64)
			#data = self.sar_collection.find({'take_number' : str(take)})
			data=self.sar_collection.find_one({'take_index' : take})
			data_path=data['path']
			f=h5py.File(data_path, 'r')
			s21=f['sar_dataset']
			nprofiles=s21.shape[0]

			#nr = 2 ** int(np.ceil(np.log2(self.nfre)))
			nr=self.nfre
			B  = self.df*nr
			dr = c0 / (2*B)
			distance = np.arange(nr) * dr

			nc0 = int(self.nfre/2.0)
			nc1 = int((self.nfre+1)/2.0)

			s21_arr=np.zeros(s21.shape, dtype = complex)
			s21_arr[:,0:nc1] = s21[:,nc0:self.nfre]
			s21_arr[:,self.nfre-nc0:self.nfre] = s21[:,0:nc0]

			reflectivity=np.absolute(ifft(s21_arr))
			reflectivity=10*np.log10(reflectivity)
			data_profiles_folder=os.path.join(results_folder, 'take_{}'.format(take))

			if not os.path.exists(data_profiles_folder):
				os.makedirs(data_profiles_folder)

			for profile in range(nprofiles):
				fig = plt.figure(1)
				im = plt.plot(distance, reflectivity[profile])
				plt.xlabel('Range (m)', fontsize = 14)
				plt.ylabel('Relative Radar Reflectivity (dB)', fontsize = 14)
				plt.savefig(os.path.join(data_profiles_folder, 'data_profile_{}.png'.format(profile)))
				fig.clear()

	def process_single_file(self, xi, xf, yi, yf, dx, dy, ifft_fact=8, win=False, algorithm="terrain_mapping"):
		self.xi = xi
		self.xf = xf
		self.yi = yi
		self.yf = yf
		self.dx = dx
		self.dy = dy
		self.algorithm=algorithm

		tmp=h5py.File(self.single_file, 'r+')
		dset=tmp['sar_dataset']
		s21=dset[...]
		self.beam_angle=float(dset.attrs['beam_angle'])
		self.xai=float(dset.attrs['xi'])
		self.xaf=float(dset.attrs['xf'])
		self.dax=float(dset.attrs['dx'])
		self.nx=int(dset.attrs['npos'])
		self.fre_min=float(dset.attrs['fi'])
		self.fre_max=float(dset.attrs['ff'])
		self.nfre=float(dset.attrs['nfre'])
		self.datetime=dset.attrs['datetime']
		self.fre_c=(self.fre_min+self.fre_max) / 2.0
		self.df=(self.fre_max - self.fre_min) / (self.nfre - 1.0)
		tmp.close()

		if self.algorithm == "terrain_mapping":
			self.nposx=int(np.ceil((self.xf-self.xi)/self.dx)+1) #number of positions axis x
			self.nposy=int(np.ceil((self.yf-self.yi)/self.dy)+1) #number of positions axis y
			self.xf=self.xi+self.dx*(self.nposx-1) #recalculating x final position
			self.yf=self.yi+self.dy*(self.nposy-1) #recalculating y final position
			self.npos=self.nposx*self.nposy #total of positions
			self.nr=2**int(np.ceil(np.log2(self.nfre*ifft_fact))) #calculate a number of ranges,
																#considering the zero padding
			n=np.arange(self.nr) #final number of ranges
			B=self.df*self.nr #final bandwidth
			dr=c0/(2*B) #recalculate resolution, considering ifft_fact
			self.rn=dr*n #range vector
			self.R=dr*self.nr #for the period verb for the interp

			xa=self.xai+self.dax*np.arange(self.nx) #antenna positions vector
			xn=self.xi+self.dx*np.arange(self.nposx) #grid vector axis x
			yn=self.yi+self.dy*np.arange(self.nposy) #grid vector axis y
			Rnk=self.calculate_Rnk(xn, yn ,xa) #vector of distance from the antenna positions to the grid

			if win:
				s21=s21*np.hanning(s21.shape[1])
				s21=s21*np.hanning(s21.shape[0])[:,np.newaxis]

			dt=datetime.strptime(self.datetime, "%d-%m-%y %H:%M:%S")
			date=str(dt.date())
			time=str(dt.time().strftime("%H:%M:%S"))

			self.tm_algorithm(s21=s21, Rnk=Rnk, take=None, index=None, date=date, time=time,
							  results_folder=None, results_collection=None)

	def process_data(self, xi, xf, yi, yf, dx, dy, R0=0.0, ifft_fact=8, win=False, algorithm="terrain_mapping"):
		#grid extension: [(xi, xf), (yi, yf)]
		#grid resolution: dx and dy
		self.xi = xi
		self.xf = xf
		self.yi = yi
		self.yf = yf
		self.dx = dx
		self.dy = dy
		self.algorithm=algorithm
		"""
		self.ifft_fact=ifft_fact
		self.window=win
		self.algorithm=algorithm
		"""

		results_folder=os.path.join("/home/andre/sar_processed_data/imaging/", self.collection_name)

		if not os.path.exists(results_folder):
			os.makedirs(results_folder)

		processed_data_db=self.db_client['sar_processed_data']
		sar_processed_data_collection=processed_data_db[self.collection_name]

		if self.algorithm == "range_migration":
			range_res=self.nfre
			cross_range_res=2**11
			rows_up=int((max(cross_range_res, self.nx)-self.nx)/2)
			#cols_right=int((max(range_res, self.nfre)-self.nfre)/2)

			rows_down=(rows_up+1) if (self.nx%2!=0 and rows_up!=0) else (rows_up)
			#cols_left=(cols_right+1) if (self.nfre%2!=0 and cols_right!=0) else (cols_right)

			#rs=(self.yf+self.yi)/2.0
			#rs=200.0
			#phi= self.calculate_matched_filter(rs, range_res, cross_range_res)
			#s0=np.exp(1j*rs*phi)

			take=3
			data=self.sar_collection.find_one({'take_index' : take})
			data_path=data['path']
			f=h5py.File(data_path, 'r')

			s21=f['sar_dataset']
			s21=self.hanning(s21) if win else s21
			#s21=np.pad(s21,[[rows_up,rows_down],[cols_left,cols_right]], 'constant', constant_values=0)
			s21_angle=np.angle(s21)
			s21=np.pad(s21,[[rows_up,rows_down],[0,0]], 'constant', constant_values=0)
			S21=np.fft.fftshift(np.fft.fft(s21,axis=0), axes=0)

			Kr=np.linspace((4*np.pi/c0)*self.fre_min, (4*np.pi/c0)*self.fre_max, self.nfre)
			Kx=np.linspace((-np.pi/self.dax), (np.pi/self.dax), cross_range_res)

			Ky=np.zeros((cross_range_res, range_res), dtype=float)
			Ky_even=np.linspace(650, 653, 2048)
			S_st=np.zeros((cross_range_res, len(Ky_even)), dtype=np.complex64)

			for row in xrange(cross_range_res):
				Ky[row,:]=np.sqrt(Kr**2-Kx[row]**2)
				#S_st[row,:]=np.interp(Ky[row,:], S21[row,:], Ky_even)
				S_st[row,:]=np.interp(Ky_even, Ky[row,:], np.real(S21[row,:]))+1j*np.interp(Ky_even, Ky[row,:], np.imag(S21[row,:]))

			H=np.zeros(2048, dtype=float)

			for row in xrange(2048):
				H[row]=0.5+0.5*np.cos(2*np.pi*(row-2048.0/2.0)/2048.0)

			for row in xrange(np.shape(S_st)[0]):
				S_st[row,:]=S_st[row,:]*H

			s_st=np.flipud(np.transpose(np.fft.ifft2(S_st, [2*np.shape(S_st)[0], 2*np.shape(S_st)[1]])))
			#s_st=np.flipud(np.transpose(np.fft.ifft2(S_st)))

			fig=plt.figure(1)
			S21_img=20*np.log10(np.abs(s_st))
			#S21_img=(np.abs(S_st))
			#S21_angle=np.angle(S21)
			#s21_angle=np.angle(s21)
			max_range=self.nfre*c0/(2*(self.fre_max-self.fre_min))
			r_index1=int((20.0/max_range)*np.shape(s_st)[1])
			r_index2=int((520.0/max_range)*np.shape(s_st)[1])
			cr_index1=int(((-150+cross_range_res*self.dax/2.0)/(cross_range_res*self.dax))*np.shape(s_st)[0])
			cr_index2=int(((150+cross_range_res*self.dax/2.0)/(cross_range_res*self.dax))*np.shape(s_st)[0])

			im=plt.imshow(S21_img[r_index1:r_index2,cr_index1:cr_index2], cmap = 'jet', aspect = 'auto', vmin=-120)
			#im=plt.imshow(20*np.log10(np.abs(S21)), cmap = 'jet', aspect = 'auto')
			#plt.imshow(self.masked_values, cmap = 'Greys', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf], vmin = self.vmin, vmax = self.vmax, interpolation = 'none')
			cbar=plt.colorbar(im, orientation = 'vertical')
			#im=plt.imshow(s21_angle, cmap = 'gray', aspect = 'auto')
			#cbar=plt.colorbar(im, orientation = 'vertical')
			plt.show()
			#plt.ylabel('Range (m)', fontsize = 14)
			#plt.xlabel('Cross-range (m)', fontsize = 14)
			#plt.savefig(images_results_folder + '/image%d.png' %take)
			#fig.clear()

			"""
			for take in range(1, self.ntakes + 1):
				print "Processing %d out of %d." %(take, self.ntakes)

				data=self.sar_collection.find_one({'take_index' : take})
				data_path=data['path']
				f=h5py.File(data_path, 'r')

				s21=f['sar_dataset']
				s21=self.hanning(s21) if win else s21
				#s21=np.pad(s21,[[rows_up,rows_down],[cols_left,cols_right]], 'constant', constant_values=0)
				s21=np.pad(s21,[[rows_up,rows_down],[0,0]], 'constant', constant_values=0)

				dt=datetime.strptime(data['datetime'], "%d-%m-%y %H:%M:%S")
				date=str(dt.date())
				time=str(dt.time().strftime("%H:%M:%S"))

				self.rm_algorithm(s21, s0, take, date, time, results_folder, phi)
			"""

		if self.algorithm == "terrain_mapping":
			#query=sar_processed_data_collection.find({"algorithm":self.algorithm})

			self.nposx=int(np.ceil((xf-xi)/dx)+1) #number of positions axis x
			self.nposy=int(np.ceil((yf-yi)/dy)+1) #number of positions axis y
			self.xf=self.xi+self.dx*(self.nposx-1) #recalculating x final position
			self.yf=self.yi+self.dy*(self.nposy-1) #recalculating y final position
			self.npos=self.nposx*self.nposy #total of positions
			self.nr=2**int(np.ceil(np.log2(self.nfre*ifft_fact))) #calculate a number of ranges,
																#considering the zero padding
			n=np.arange(self.nr) #final number of ranges
			B=self.df*self.nr #final bandwidth
			dr=c0/(2*B) #recalculate resolution, considering ifft_fact
			self.rn=dr*n #range vector
			self.R=dr*self.nr #for the period verb for the interp

			xa=self.xai+self.dax*np.arange(self.nx) #antenna positions vector
			xn=self.xi+self.dx*np.arange(self.nposx) #grid vector axis x
			yn=self.yi+self.dy*np.arange(self.nposy) #grid vector axis y

			#sar_processed_data_collection.delete_many({})

			Rnk=self.calculate_Rnk(xn, yn ,xa) #vector of distance from the antenna positions to the grid
			Rnk_folder=results_folder+"/Rnk.hdf5"
			f=h5py.File(Rnk_folder, 'w')
			dset=f.create_dataset("Rnk", (Rnk.shape), dtype=np.float32)
			dset[...]=Rnk
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
			file_temp = h5py.File(parameters['Rnk_folder'], 'r+')
			dset = file_temp["Rnk"]
			Rnk = dset[...]
			print Rnk
			file_temp.close()

			for index, take in enumerate(self.ntakes_list, 1):
				print "Processing %d out of %d." %(take, self.ntakes)
				data=self.sar_collection.find_one({'take_index' : take})
				data_path=data['path']
				f=h5py.File(data_path, 'r')
				dset=f['sar_dataset']
				datetime_aux=dset.attrs['datetime']
				s21=dset[...]
				f.close()

				if win:
					s21 = s21 * np.hanning(s21.shape[1])
					s21 = s21 * np.hanning(s21.shape[0])[:,np.newaxis]

				dt = datetime.strptime(datetime_aux, "%d-%m-%y %H:%M:%S")
				date = str(dt.date())
				time = str(dt.time().strftime("%H:%M:%S"))

				self.tm_algorithm(s21=s21, Rnk=Rnk, take=take, index=index, date=date, time=time,
								  results_folder=results_folder, results_collection=sar_processed_data_collection)

	def calculate_matched_filter(self, rs, range_res, cross_range_res):
		k=(4*np.pi/c0)*np.linspace(self.fre_min, self.fre_max, range_res)
		ku=(np.pi/self.dax)*np.linspace(-1, 1, cross_range_res)
		kk, kuu = np.meshgrid(k, ku)
		#phi=np.nan_to_num(rs*np.sqrt(kk**2-kuu**2))
		#phi=np.nan_to_num(np.sqrt(kk**2-kuu**2))
		phi=np.sqrt(kk**2-kuu**2)
		"""
		self.kxmn = np.sqrt(kk**2-kuu**2)
		self.kxmn = np.nan_to_num(self.kxmn)
		self.kymin = np.amin(ku)
		self.kymax = np.amax(ku)
		self.kxmin = np.amin(self.kxmn)
		self.kxmax = np.amax(self.kxmn)
		print self.kxmin, self.kxmax
		"""

		#return np.exp(1j*phi)
		return phi

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

	def tm_algorithm(self, s21, Rnk, date, time, take=None, index=None, results_folder=None, results_collection=None):
		I = np.zeros([self.npos], dtype = np.complex64)
		s21_arr = np.zeros([self.nx, self.nr], dtype = np.complex64)
		nc0 = int(self.nfre/2.0) #first chunk of the frequency: f0,fc
		nc1 = int((self.nfre+1)/2.0) #first chunk of the frequency: fc,ff
		s21_arr[:,0:nc1] = s21[:, nc0:self.nfre] #invert array order
		s21_arr[:,self.nr - nc0: self.nr] = s21[:, 0:nc0]
		Fn0 = self.nr * ifft(s21_arr, n = self.nr)


		for k in range(0,self.nx):
			Fn=np.interp(Rnk[k,:] - R0, self.rn, np.real(Fn0[k,:])) + 1j * np.interp(Rnk[k,:] - R0, self.rn, np.imag(Fn0[k,:]))
			Fn*=np.exp(4j * np.pi * (self.fre_min/c0) * (Rnk[k,:] - R0))
			I+=Fn

		I /= (self.nfre * self.nx)
		I = np.reshape(I, (self.nposy, self.nposx))
		I = np.flipud(I)

		if take==None and index==None and results_folder==None and results_collection==None:
			I=10*np.log10(np.absolute(I))
			fig=plt.figure(1)
			self.vmin=np.amin(I)+32
			self.vmax=np.amax(I)

			folder=os.path.dirname(self.single_file)
			im = plt.imshow(I, cmap = 'jet', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf], vmin = self.vmin, vmax = self.vmax)
			cbar = plt.colorbar(im, orientation = 'vertical')
			plt.ylabel('Range (m)', fontsize = 14)
			plt.xlabel('Cross-range (m)', fontsize = 14)
			plt.savefig('{}{}'.format(folder,'/img.png'))
			fig.clear()
		else:
			data_results_folder = results_folder + "/data"
			images_results_folder = results_folder + "/images"

			if not os.path.exists(data_results_folder):
				os.makedirs(data_results_folder)

			if not os.path.exists(images_results_folder):
				os.makedirs(images_results_folder)

			f = h5py.File("{}/image{}.hdf5".format(data_results_folder, take), 'w')
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
					"route": data_results_folder + "/image{}.hdf5".format(take)}

			results_collection.insert(post)
			post = None

			I = 10 * np.log10(np.absolute(I))
			fig = plt.figure(1)

			if index==1:
				self.vmin = np.amin(I)+32
				#self.vmin = np.amin(I)
				self.vmax = np.amax(I)

				'''
				aux = int((self.yf - (self.xf * np.tan(self.beam_angle * np.pi /180.0))) * I.shape[0] / (self.yf - self.yi))
				mask = np.zeros(I.shape)

				count = 0

				for k in range(self.nposy):
					#if k >= (int(0.5 * self.nposx / np.tan(32.0 * np.pi /180.0)) + 1):
					if k >= (aux + 1):
						mask[k, 0:count] = 1
						mask[k, self.nposx - count -1:self.nposx-1] = 1
						count = count + 1
				self.masked_values = np.ma.masked_where(mask == 0, mask)
				'''

			im = plt.imshow(I, cmap = 'jet', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf], vmin = self.vmin, vmax = self.vmax)
			#plt.imshow(self.masked_values, cmap = 'Greys', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf], vmin = self.vmin, vmax = self.vmax, interpolation = 'none')
			cbar = plt.colorbar(im, orientation = 'vertical')
			plt.ylabel('Range (m)', fontsize = 14)
			plt.xlabel('Cross-range (m)', fontsize = 14)
			plt.savefig(images_results_folder + '/image%d.png' %take)
			fig.clear()

	def rm_algorithm(self, s21, s0, take, date, time, results_folder, phi):
		Fmn=np.fft.fftshift(fft(s21,axis=0))*s0

		fig=plt.figure(1)
		plt.imshow()
		plt.show()
		fig.clear()

		'''
		ky_len=len(s0)
		res = 2**12
		kstart, kstop = np.amax(phi), np.amin(phi)
		kx_even = np.linspace(kstart, kstop, res)
		kx=phi
		Fst = np.zeros((len(kx), len(kx_even)), dtype = np.complex64)

		for i in range(len(kx)):
			interp_fn = interp1d(kx[i], Fmn[i], bounds_error = False, fill_value=0)
			Fst[i] = interp_fn(kx_even)

		Fst=np.nan_to_num(Fst)
		ifft_len=[4*len(Fst), 4*len(Fst[0])]
		f=(np.rot90(ifft2(Fst, ifft_len)))

		#max_range = 2 * np.pi * f.shape[0]/(self.kxmax - self.kxmin)
		#max_crange = 2 * np.pi * f.shape[1]/(self.kymax - self.kymin)
		#print max_range, max_crange

		I=20*np.log10(np.absolute(f))
		fig=plt.figure(1)
		(self.vmin, self.vmax)=(np.amin(I), np.amax(I)) if take==1 else (self.vmin, self.vmax)

		images_results_folder=results_folder+"/images"

		im = plt.imshow(I, cmap = 'jet', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf], vmin = self.vmin, vmax = self.vmax)
		cbar = plt.colorbar(im, orientation = 'vertical')
		plt.ylabel('Range (m)', fontsize = 14)
		plt.xlabel('Cross-range (m)', fontsize = 14)
		#plt.show()
		plt.savefig(images_results_folder + '/image%d.png' %take)
		fig.clear()
		'''

class jro_sliding_processor():
	def __init__(self, db_name, collection_name):
		self.db_name=db_name
		self.collection_name = collection_name
		self.db_client = MongoClient() #connect to MongoClient
		self.results_path=SLIDING_RESULTS_PATH

		self.results_folder=os.path.join(self.results_path, self.collection_name)

		if not os.path.exists(self.results_folder):
			os.makedirs(self.results_folder)

	def read_data(self):
		db = self.db_client[self.db_name] #read 'sar_processed_data'
		self.sar_collection = db[self.collection_name]
		parameters = self.sar_collection.find({'type' : 'parameters'})[0]
		self.ntakes_list=sorted([str(item) for item in self.sar_collection.find().distinct('take')], key=int)
		self.ntakes = len(self.ntakes_list)
		print "Found {} takes to process.".format(self.ntakes-1)

		self.xi = float(parameters['xi'])
		self.xf = float(parameters['xf'])
		self.yi = float(parameters['yi'])
		self.yf = float(parameters['yf'])
		self.fre_min = float(parameters['fi'])
		self.fre_max = float(parameters['ff'])
		self.fre_c = (self.fre_min+self.fre_max) / 2.0
		self.lambda_d = 1000*(c0/(self.fre_c*4*np.pi))

		data=self.sar_collection.find_one({'take' : self.ntakes_list[0]})

		file_temp=h5py.File(data['route'], 'r')
		dset=file_temp["Complex_image"]
		image_sample=dset[...]
		file_temp.close()
		self.image_shape=np.shape(image_sample)

	#def calculate_sliding(self, threshold, output_images):
	def calculate_sliding(self, threshold_start, threshold_stop, output_images):
		#self.threshold=threshold
		self.threshold_start=threshold_start
		self.threshold_stop=threshold_stop

		print "Processing data..."

		correlation_file=os.path.join(self.results_folder, "correlation.hdf5")

		if not os.path.exists(correlation_file):
			correlation=self.calculate_correlation()
			file_temp=h5py.File(correlation_file, 'w')
			dset=file_temp.create_dataset("Correlation", (correlation.shape), dtype=np.complex64)
			dset[...]=correlation
			file_temp.close()
		else:
			file_temp=h5py.File(correlation_file, 'r+')
			dset=file_temp["Correlation"]
			correlation=dset[...]
			file_temp.close()

		#aux_x=int((self.yi*np.tan(37.5*np.pi/180.0))*correlation.shape[1]/(self.xf-self.xi))
		aux_y=int((self.yf-((self.xf-self.yi*np.tan(37.5*np.pi/180.0))*np.tan(52.5*np.pi/180.0)))*correlation.shape[0]/(self.yf-self.yi))

		#aux=int((self.yf-(self.xf*np.tan(52.5*np.pi/180.0)))*correlation.shape[0]/(self.yf-self.yi))
		mask_aux=np.ones(correlation.shape)
		nposx=correlation.shape[1]
		nposy=correlation.shape[0]
		count=1

		for k in range(nposy):
			if k>=(aux_y+1):
				npixels=int((count*((yf-yi)/correlation.shape[0])*np.tan(37.5*np.pi/180.0))*correlation.shape[1]/(self.xf-self.xi))
				mask_aux[k, 0:npixels]=0
				mask_aux[k, nposx-npixels-1:nposx-1]=0
				count=count+1
		masked_values=np.ma.masked_where(mask_aux==1, mask_aux)

		fig = plt.figure(1)
		plt.title("Complex correlation magnitude", fontsize = 11)
		im = plt.imshow(np.absolute(correlation)*mask_aux, cmap = 'jet', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf] , vmin = 0, vmax = 1)
		plt.imshow(masked_values, cmap = 'Greys', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])
		plt.colorbar(im, orientation = 'vertical', format='%.1f')
		plt.savefig(self.results_folder + "/complex_correlation_mag.png")
		fig.clear()

		mask=np.absolute(correlation)*mask_aux
		mask_binary=np.ones(np.shape(mask))
		[mask_binary[mask<=threshold_start], mask_binary[mask>=threshold_stop]]=[0,0]
		masked_values=np.ma.masked_where(mask_binary==0, mask)

		#vmax, vmin=np.pi*self.lambda_d*np.array([1,-1])
		vmax, vmin=np.pi*np.array([1,-1])
		mag_mean = np.zeros(self.ntakes - 1)
		phase_mean = np.zeros(self.ntakes - 1)
		phase_mean_sum = np.zeros(self.ntakes - 1)
		phase_std_dev = np.zeros(self.ntakes - 1)
		date_values=[]

		fig = plt.figure(1)

		file_temp=h5py.File(self.results_folder+"/mask.hdf5", 'w')
		dset=file_temp.create_dataset("Mask", (mask.shape), dtype = np.uint8)
		dset[...]=mask_binary
		file_temp.close()
		width=len(str(self.ntakes))
		print "Processing with threshold limits: [{}, {}]".format(threshold_start,threshold_stop)

		for index, take in enumerate(self.ntakes_list,0):
			if index==len(self.ntakes_list)-1:
				break
			data=self.sar_collection.find_one({'take' : self.ntakes_list[index]})

			file_temp=h5py.File(data['route'], 'r')
			dset=file_temp["Complex_image"]
			Imagen_master=dset[...]
			file_temp.close()

			data=self.sar_collection.find_one({'take' : self.ntakes_list[index+1]})

			date=(data['date'])
			time=(data['time'])
			file_temp=h5py.File(data['route'], 'r')
			dset=file_temp["Complex_image"]
			Imagen_slave=dset[...]
			file_temp.close()

			phase=np.angle(Imagen_master*np.conj(Imagen_slave))
			magnitude=np.absolute(Imagen_master)
			masked_angle=np.ma.masked_where(mask_binary==0, phase)
			masked_magnitude=np.ma.masked_where(mask_binary==0, magnitude)

			mag_mean[index] = masked_magnitude.mean()
			phase_mean[index] = masked_angle.mean()*self.lambda_d
			phase_mean_sum[index] = np.sum(phase_mean)
			phase_std_dev[index] = np.std(masked_angle*self.lambda_d)
			print "indice {}: {} {} {} {}".format(index, mag_mean[index], phase_mean[index], phase_mean_sum[index], phase_std_dev[index])
			date_values.append(datetime.strptime(''.join((date, time)), ''.join((date_format, time_format)))+timedelta(days=7))

			#if index>=1:
			#	print (date_values[index]-date_values[index-1]).total_seconds()
			#if index==0:
			#	vmax=np.amax(np.absolute(Imagen_master*np.conj(Imagen_slave)))
			#	vmin=np.amin(np.absolute(Imagen_master*np.conj(Imagen_slave)))

			if output_images:
				if index==0:
					im=plt.imshow(self.lambda_d*phase, cmap = 'jet', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf], vmin=vmin, vmax=vmax)
				plt.suptitle("Image {} and {}".format(index, index+1))
				plt.ylabel('Range (m)', fontsize = 14)
				plt.xlabel('Cross-range (m)', fontsize = 14)
				cbar=plt.colorbar(im, orientation = 'vertical', format='%.2f')
				#plt.imshow(self.lambda_d*phase, cmap = 'jet', aspect = 'auto',	extent = [self.xi,self.xf,self.yi,self.yf], vmin=vmin, vmax=vmax)
				plt.imshow(phase, cmap = 'jet', aspect = 'auto',	extent = [self.xi,self.xf,self.yi,self.yf], vmin=vmin, vmax=vmax)
				#plt.imshow(np.absolute(Imagen_master*np.conj(Imagen_slave)), cmap = 'jet', aspect = 'auto',	extent = [self.xi,self.xf,self.yi,self.yf], vmin=vmin, vmax=vmax)
				#plt.imshow(masked_values, cmap = 'Greys', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])
				#plt.savefig(os.path.join(self.results_folder, "take_{}_{}.png".format("{0:0{width}}".format(take, width=width), "{0:0{width}}".format(take+1, width=width))))
				plt.savefig(os.path.join(self.results_folder, "take_{}_{}.png".format(index, index+1)))
				plt.close()

		#for i in range(1,len(date_values)):
		#	delta=date_values[i]-date_values[i-1]
		#	if delta.total_seconds()>=datetime.timedelta(minutes=17).total_seconds():

		fig.clear()
		fig = plt.figure(figsize = (10.0, 6.0))

		plt.subplot(221)
		plt.title(r'$\overline{\Delta r}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\overline{\Delta r}\/\/(mm)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		#plt.plot(date_values, phase_mean,'bo', markersize=2)
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
		#plt.plot(date_values, phase_mean_sum,'bo', markersize=2)
		plt.plot(date_values, phase_mean_sum)
		ax = plt.gca()
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels, rotation=30, fontsize=10)

		sub = plt.subplot(223)
		plt.title(r'$\sigma_{\Delta r}\/\/vs\/\/time$', fontsize = 16)
		plt.ylabel(r'$\sigma_{\Delta r}\/\/(mm)$', fontsize = 16)
		plt.xlabel(r'$time$', fontsize = 16)
		#plt.plot(date_values, phase_std_dev,'bo',markersize=2)
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
		#plt.plot(date_values, mag_mean,'bo', markersize=2)
		plt.plot(date_values, mag_mean)
		ax = plt.gca()
		ax.set_ylim([0.0 , np.amax(mag_mean) * 1.2])
		ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
		labels = ax.get_xticklabels()
		plt.setp(labels,  rotation=30, fontsize=10)

		plt.tight_layout()
		plt.savefig(os.path.join(self.results_folder,
					"statistical_report_th_{}_{}.png".
					format(str(threshold_start).replace(".","u"),
					str(threshold_stop).replace(".", "u"))))
		plt.close()
		print "Done!"

	def calculate_correlation(self):
		for index, take in enumerate(self.ntakes_list,0):
			if index==len(self.ntakes_list)-1:
				break

			data = self.sar_collection.find_one({'take' : self.ntakes_list[index]})
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_master = dset[...]
			file_temp.close()

			data = self.sar_collection.find_one({'take' : self.ntakes_list[index+1]})
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_slave = dset[...]
			file_temp.close()

			if index==0:
				num=Imagen_master*np.conj(Imagen_slave)
				den=np.sqrt(np.absolute(Imagen_master)**2 * np.absolute(Imagen_slave)**2)
				continue
			num += Imagen_master * np.conj(Imagen_slave)
			den += np.sqrt(np.absolute(Imagen_master)**2 * np.absolute(Imagen_slave)**2)

		return num/den

	def clean_data_report(self, threshold):
		self.clean_data_list=[]
		file_temp=h5py.File(self.results_folder+"/mask.hdf5", 'r')
		dset=file_temp["Mask"]
		mask=dset[...]
		file_temp.close()
		clean_data_set=[]

		for index, take in enumerate(self.ntakes_list,0):
			if index==len(self.ntakes_list)-1:
				break
			data = self.sar_collection.find_one({'take' : self.ntakes_list[index]})
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_master = dset[...]
			file_temp.close()

			data = self.sar_collection.find_one({'take' : self.ntakes_list[index+1]})
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_slave = dset[...]
			file_temp.close()

			phase=np.angle(Imagen_master*np.conj(Imagen_slave))
			masked_angle = np.ma.masked_where(mask==0, phase)
			phase_std_dev=np.std(masked_angle)*self.lambda_d

			if phase_std_dev<threshold:
				clean_data_set.append(self.ntakes_list[index])

		self.clean_data_list=sorted(list(clean_data_set), key=int)

		clean_results_folder = self.results_folder + "/clean_data"
		if not os.path.exists(clean_results_folder):
			os.makedirs(clean_results_folder)

		report_len=len(self.clean_data_list)-1
		mag_mean=np.zeros(report_len)
		phase_mean=np.zeros(report_len)
		phase_mean_sum=np.zeros(report_len)
		phase_std_dev=np.zeros(report_len)
		date_values=[]

		fig = plt.figure(1)
		vmax, vmin=np.pi*self.lambda_d*np.array([1,-1])

		for index, element in enumerate(self.clean_data_list, 0):
			if index==len(self.clean_data_list)-1:
				break
			master=int(self.clean_data_list[index])
			slave=int(self.clean_data_list[index + 1])

			print "Processing take {} and {}.".format(master, slave)

			data = self.sar_collection.find_one({'take' : self.ntakes_list[master]})
			file_temp = h5py.File(data['route'], 'r')
			dset = file_temp["Complex_image"]
			Imagen_master = dset[...]
			file_temp.close()

			data = self.sar_collection.find_one({'take' : self.ntakes_list[slave]})
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
		file_temp=h5py.File(self.results_folder+"/mask.hdf5", 'r')
		dset=file_temp["Mask"]
		mask=dset[...]
		file_temp.close()

		n=len(self.clean_data_list)-(len(self.clean_data_list)%integration_factor)
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
		mask_path=os.path.join(self.results_folder, 'mask.hdf5')
		file_temp=h5py.File(mask_path, 'r')
		dset=file_temp["Mask"]
		mask=dset[...]
		file_temp.close()

		new_clean_data_list = []

		for index, element in enumerate(self.clean_data_list):
			if index == len(self.clean_data_list) - 1:
				break
			new_clean_data_list.append([self.clean_data_list[index], self.clean_data_list[index + 1]])

		n = len(new_clean_data_list) - (len(new_clean_data_list) % integration_factor)
		new_clean_data_list = new_clean_data_list[:n]
		groups_list = [new_clean_data_list[i: i + integration_factor] for i in range(0, len(new_clean_data_list), integration_factor)]

		integrated_sliding2_results_folder=os.path.join(self.results_folder, "integrated_sliding2_factor_{}".format(integration_factor))

		if not os.path.exists(integrated_sliding2_results_folder):
			os.makedirs(integrated_sliding2_results_folder)

		integrated_sliding2_len = len(groups_list) - 1
		mag_mean = np.zeros(integrated_sliding2_len)
		phase_mean = np.zeros(integrated_sliding2_len)
		phase_mean_sum = np.zeros(integrated_sliding2_len)
		phase_std_dev = np.zeros(integrated_sliding2_len)
		date_values = []
		width=len(str(len((groups_list))))

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

				if enum==0:
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
							mask_aux[k, nposx-count-1:nposx-1] = 1
							count = count + 1
					masked_values = np.ma.masked_where(mask_aux == 0, mask_aux)
					plt.imshow(masked_values, cmap = 'binary', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])
					plt.imshow(masked_plot, cmap = 'binary', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])

				if enum > 0:
					im = plt.imshow(self.lambda_d * phase, cmap = 'jet', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf], vmin = vmin, vmax = vmax)
					plt.imshow(masked_values, cmap = 'binary', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])
					plt.imshow(masked_plot, cmap = 'binary', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])

				plt.savefig(os.path.join(integrated_sliding2_results_folder, "groups_{}_{}.png".format("{0:0{width}}".format(enum, width=width), "{0:0{width}}".format(enum+1, width=width))))

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

	def calculate_sliding_split_threshold(self, threshold_number, output_images=False):
		threshold_bins=np.linspace(0,1,threshold_number+1)

		for n in range(len(threshold_bins)-1):
			threshold_start=threshold_bins[n]
			threshold_stop=threshold_bins[n+1]
			self.calculate_sliding(threshold_start, threshold_stop, output_images)

	def calculate_parameters_grid(self, nx=3, ny=3):
		print "Processing data..."

		#aux_x=int((self.yi*np.tan(37.5*np.pi/180.0))*correlation.shape[1]/(self.xf-self.xi))
		aux_y=int((self.yf-((self.xf-self.yi*np.tan(37.5*np.pi/180.0))*np.tan(52.5*np.pi/180.0)))*self.image_shape[0]/(self.yf-self.yi))

		#aux=int((self.yf-(self.xf*np.tan(52.5*np.pi/180.0)))*correlation.shape[0]/(self.yf-self.yi))
		mask_binary=np.ones(self.image_shape)
		nposx=self.image_shape[1]
		nposy=self.image_shape[0]
		count=1

		for k in range(nposy):
			if k>=(aux_y+1):
				npixels=int((count*((yf-yi)/self.image_shape[0])*np.tan(37.5*np.pi/180.0))*self.image_shape[1]/(self.xf-self.xi))
				mask_binary[k, 0:npixels]=0
				mask_binary[k, nposx-npixels-1:nposx-1]=0
				count=count+1

		vmax, vmin=np.pi*self.lambda_d*np.array([1,-1])

		mag_mean=np.zeros((nx, ny, self.ntakes-1))
		phase_mean=np.zeros((nx, ny, self.ntakes-1))
		phase_mean_sum=np.zeros((nx, ny, self.ntakes-1))
		phase_std_dev=np.zeros((nx, ny, self.ntakes-1))

		date_values_str=[]

		nominal_width_x=int(self.image_shape[1]/nx)
		nominal_width_y=int(self.image_shape[0]/ny)

		"""
		for nx_aux in range(nx):
			for ny_aux in range(ny):
				y, x=np.ogrid[0:self.image_shape[0], 0:self.image_shape[1]]
				xv, yv=np.meshgrid(np.logical_and(x>(nx_aux)*nominal_width_x,x<(nx_aux+1)*nominal_width_x), np.logical_and(y>(ny_aux)*nominal_width_y,y<(ny_aux+1)*nominal_width_y))
				mask_array=np.logical_and(np.logical_and(xv,yv), mask_binary)
				#print np.any(mask_array)
				fig=plt.figure(1)
				plt.imshow(mask_array, cmap = 'Greys', aspect = 'auto', extent = [self.xi,self.xf,self.yi,self.yf])
				plt.show()
				fig.clear()
		"""

		print "Start time: {}".format(datetime.now())

		patch_folder=os.path.join(self.results_folder, "segmented_image_{}_{}".format(nx, ny))
		if not os.path.exists(patch_folder):
			os.makedirs(patch_folder)

		if not os.path.exists(os.path.join(patch_folder, "statistical_descriptors.hdf5")):
			for index, take in enumerate(self.ntakes_list,0):
				print "Processing take {}".format(index)
				if index==len(self.ntakes_list)-1:
					break
				data=self.sar_collection.find_one({'take' : self.ntakes_list[index]})

				file_temp=h5py.File(data['route'], 'r')
				dset=file_temp["Complex_image"]
				Imagen_master=dset[...]
				file_temp.close()

				data=self.sar_collection.find_one({'take' : self.ntakes_list[index+1]})

				date=(data['date'])
				time=(data['time'])
				file_temp=h5py.File(data['route'], 'r')
				dset=file_temp["Complex_image"]
				Imagen_slave=dset[...]
				file_temp.close()
				grid_mask=np.zeros(self.image_shape)

				for nx_aux in range(nx):
					for ny_aux in range(ny):
						y, x=np.ogrid[0:self.image_shape[0], 0:self.image_shape[1]]
						xv, yv=np.meshgrid(np.logical_and(x>(nx_aux)*nominal_width_x,x<(nx_aux+1)*nominal_width_x), np.logical_and(y>(ny_aux)*nominal_width_y,y<(ny_aux+1)*nominal_width_y))
						mask_array=np.logical_and(np.logical_and(xv,yv), mask_binary)

						phase=np.angle(Imagen_master*np.conj(Imagen_slave))
						magnitude=np.absolute(Imagen_master)

						masked_angle=np.ma.masked_where(mask_array==0, phase)
						masked_magnitude=np.ma.masked_where(mask_array==0, magnitude)

						if np.any(mask_array):
							mag_mean[nx_aux, ny_aux, index] = masked_magnitude.mean()
							phase_mean[nx_aux, ny_aux, index] = masked_angle.mean()*self.lambda_d
							phase_mean_sum[nx_aux, ny_aux, index] = np.sum(phase_mean[nx_aux, ny_aux, :])
							phase_std_dev[nx_aux, ny_aux, index] = np.std(masked_angle*self.lambda_d)

						if (nx_aux==0 and ny_aux==0):
							#date_values.append(datetime.strptime(''.join((date, time)), ''.join((date_format, time_format)))+timedelta(days=7))
							date_aux=' '.join((date, time)).encode('utf8')
							date_values_str.append(date_aux)

			format=' '.join((date_format, time_format))
			f=h5py.File(os.path.join(patch_folder, "statistical_descriptors.hdf5"), 'w')
			f.attrs['format']=format
			f.attrs['date_list']=date_values_str
			dset=f.create_dataset('mag_mean', (np.shape(mag_mean)), dtype=float)
			dset[...]=mag_mean
			dset=f.create_dataset('phase_mean', (np.shape(phase_mean)), dtype=float)
			dset[...]=phase_mean
			dset=f.create_dataset('phase_mean_sum', (np.shape(phase_mean_sum)), dtype=float)
			dset[...]=phase_mean_sum
			dset=f.create_dataset('phase_std_dev', (np.shape(phase_std_dev)), dtype=float)
			dset[...]=phase_std_dev
			f.close()
		else:
			f=h5py.File(os.path.join(patch_folder, "statistical_descriptors.hdf5"), 'r')
			date_values_str=f.attrs['date_list']
			format=f.attrs['format']
			dset=f["mag_mean"]
			mag_mean=dset[...]
			dset=f["phase_mean"]
			phase_mean=dset[...]
			dset=f["phase_mean_sum"]
			phase_mean_sum=dset[...]
			dset=f["phase_std_dev"]
			phase_std_dev=dset[...]
			f.close()

		date_values=[]
		for date in date_values_str:
			date_values.append(datetime.strptime(date, format)+timedelta(days=7))

		for nx_aux in range(nx):
			for ny_aux in range(ny):
				fig = plt.figure(1)
				fig.clear()
				fig = plt.figure(figsize = (10.0, 6.0))

				plt.subplot(221)
				plt.title(r'$\overline{\Delta r}\/\/vs\/\/time$', fontsize = 16)
				plt.ylabel(r'$\overline{\Delta r}\/\/(mm)$', fontsize = 16)
				plt.xlabel(r'$time$', fontsize = 16)
				#plt.plot(date_values, phase_mean,'bo', markersize=2)
				plt.plot(date_values, phase_mean[nx_aux, ny_aux, :])
				ax = plt.gca()
				ax.set_ylim([-(np.amax(phase_mean[nx_aux, ny_aux, :]) * 2.0), (np.amax(phase_mean[nx_aux, ny_aux, :]) * 2.0)])
				ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
				labels = ax.get_xticklabels()
				plt.setp(labels, rotation=30, fontsize=10)

				plt.subplot(222)
				plt.title(r'$\overline{\Delta r_{acc}}\/\/(mm)\/\/vs\/\/time$', fontsize = 16)
				plt.ylabel(r'$\overline{\Delta r_{acc}}$', fontsize = 16)
				plt.xlabel(r'$time$', fontsize = 16)
				#plt.plot(date_values, phase_mean_sum,'bo', markersize=2)
				plt.plot(date_values, phase_mean_sum[nx_aux, ny_aux, :])
				ax = plt.gca()
				ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
				labels = ax.get_xticklabels()
				plt.setp(labels, rotation=30, fontsize=10)

				sub = plt.subplot(223)
				plt.title(r'$\sigma_{\Delta r}\/\/vs\/\/time$', fontsize = 16)
				plt.ylabel(r'$\sigma_{\Delta r}\/\/(mm)$', fontsize = 16)
				plt.xlabel(r'$time$', fontsize = 16)
				#plt.plot(date_values, phase_std_dev,'bo',markersize=2)
				plt.plot(date_values, phase_std_dev[nx_aux, ny_aux, :])
				ax = plt.gca()
				ax.set_ylim([0.0 , np.amax(phase_std_dev[nx_aux, ny_aux, :]) * 1.2])
				ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
				labels = ax.get_xticklabels()
				plt.setp(labels, rotation=30, fontsize=10)

				plt.subplot(224)
				plt.title(r'$\overline{mag}\/\/vs\/\/time$', fontsize = 16)
				plt.ylabel(r'$\overline{mag}$', fontsize = 16)
				plt.xlabel(r'$time$', fontsize = 16)
				#plt.plot(date_values, mag_mean,'bo', markersize=2)
				plt.plot(date_values, mag_mean[nx_aux, ny_aux, :])
				ax = plt.gca()
				ax.set_ylim([0.0 , np.amax(mag_mean[nx_aux, ny_aux, :]) * 1.2])
				ax.xaxis.set_major_formatter(DateFormatter('%d/%m %H:%M'))
				labels = ax.get_xticklabels()
				plt.setp(labels,  rotation=30, fontsize=10)

				plt.tight_layout()

				plt.savefig(os.path.join(patch_folder,
							"statistical_report_patch_{}_{}.png".
							format(nx_aux, ny_aux)))
				plt.close()
				print "Done!"
		print "Finish time: {}".format(datetime.now())

if __name__ == "__main__":
	xi=-350.0
	xf=350.0
	yi=100.0
	yf=950.0

	R0=0.0
	dx=0.5
	dy=0.5
	collection_name = 'san_mateo_09-10-18_03:50:12'

	"""
	db_name='sar-raw-data'
	algorithm='terrain_mapping'
	dp=jro_gbsar_processor(db_name=db_name, collection_name=collection_name)
	#dp=jro_gbsar_processor(single_file='/home/andre/sar_raw_data/san_mateo_09-10-18_03:50:12/dset_1.hdf5')
	#dp.process_single_file(xi=xi, xf=xf, yi=yi, yf=yf, dx=dx, dy=dy, ifft_fact=8, win=True, algorithm=algorithm)
	dp.insert_data_db()
	dp.read_data()
	dp.process_data(xi=xi, xf=xf, yi=yi, yf=yf, dx=dx, dy=dy, ifft_fact=8, win=True, algorithm=algorithm)
	#dp.plot_data_profiles()

	"""
	db_name='sar_processed_data'
	x=jro_sliding_processor(db_name=db_name, collection_name=collection_name)
	x.read_data()
	#x.calculate_parameters_grid(nx=8, ny=8)
	x.calculate_sliding(threshold_start=0.85, threshold_stop=1, output_images=True)
	#x.calculate_sliding_split_threshold(threshold_number=9)
	#x.clean_data_report(1.25)
	#x.calculate_integrated_sliding2(integration_factor=2, output_images=False)
	#x.calculate_integrated_sliding2(integration_factor = 1)
