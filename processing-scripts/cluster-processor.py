import numpy as np

from scipy.fftpack import ifft, fft, ifft2, fftshift
from scipy import interpolate
from scipy.interpolate import griddata, interp1d
from multiprocessing import Pool

import ast, time, json, unicodedata, os, datetime, copyreg, types
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid=	'ignore')
np.set_printoptions(threshold=np.nan)
date_format="%Y-%m-%d"
time_format="%H:%M:%S"
c0=299792458.0
#RAW_DATA_PATH='/data/users/jflorentino/raw_data'
#IMAGING_RESULTS_PATH='/data/users/jflorentino/processed_data/imaging'
RAW_DATA_PATH='/home/andre/Documents/andre/code/python'
IMAGING_RESULTS_PATH='/home/andre/Documents/andre/code/python/imaging'

class jro_gbsar_processor():
	def __init__(self, folder_name, xi, xf, yi, yf, dx, dy, R0=0.0, ifft_fact=8, win=False):
		self.data_folder=os.path.join(RAW_DATA_PATH, folder_name)
		self.imaging_results_folder=os.path.join(IMAGING_RESULTS_PATH, folder_name)
		self.xi=xi
		self.xf=xf
		self.yi=yi
		self.yf=yf
		self.dx=dx
		self.dy=dy
		self.ifft_fact=ifft_fact
		self.win=win
		self.R0=R0

		copyreg.pickle(types.MethodType, self._pickle_method)

	def read_metadata(self):
		f=open(os.path.join(self.data_folder, 'metadata.json'), "r")
		parameters=ast.literal_eval(f.read())
		f.close()
		self.beam_angle=parameters['beam_angle']
		self.xai=float(parameters['start_position'])
		self.xaf=float(parameters['stop_position'])
		self.dax=float(parameters['delta'])
		self.nx=int(parameters['npos'])
		self.fre_min=parameters['start_freq']
		self.fre_max=parameters['stop_freq']
		self.nfre=parameters['nfre']
		self.fre_c=(self.fre_min+self.fre_max)/2.0
		self.df=(self.fre_max - self.fre_min)/(self.nfre - 1.0)

	def calculate_parameters(self):
		self.nposx=int(np.ceil((self.xf-self.xi)/self.dx)+1) #number of positions axis x
		self.nposy=int(np.ceil((self.yf-self.yi)/self.dy)+1) #number of positions axis y
		self.xf=self.xi+self.dx*(self.nposx-1) #recalculating x final position
		self.yf=self.yi+self.dy*(self.nposy-1) #recalculating y final position
		self.npos=self.nposx*self.nposy #total of positions
		self.nr=2**int(np.ceil(np.log2(self.nfre*self.ifft_fact))) #calculate a number of ranges,
															#considering the zero padding
		self.n=np.arange(self.nr) #final number of ranges
		self.B=self.df*self.nr #final bandwidth
		self.dr=c0/(2*self.B) #recalculate resolution, considering ifft_fact
		self.rn=self.dr*self.n #range vector
		self.R=self.dr*self.nr #for the period verb for the interp

		self.xa=self.xai+self.dax*np.arange(self.nx) #antenna positions vector
		self.xn=self.xi+self.dx*np.arange(self.nposx) #grid vector axis x
		self.yn=self.yi+self.dy*np.arange(self.nposy) #grid vector axis y
		#self.xv, self.yv=np.meshgrid(self.xn, self.yn)
		#self.Rnk2=np.zeros((self.nx, self.nposy, self.nposx,), dtype=np.float32)
		#for k in range(self.nx):
		#	self.Rnk[k, :, :]=np.sqrt((self.xv-self.xa[k])**2+self.yv**2)

		self.Rnk=np.zeros((self.nx, self.npos,), dtype=np.float32)
		for k in range(self.nx):
			for y in range(self.nposy):
				self.Rnk[k, y*self.nposx:(y+1)*self.nposx]=np.sqrt((self.xn-self.xa[k])**2+self.yn[y]**2)

	def process_data_pool(self):
		if not os.path.exists(self.imaging_results_folder):
			os.makedirs(self.imaging_results_folder)

		self.calculate_parameters()
		file_list=[]

		for dirpath,_,filenames in os.walk(self.data_folder):
			for f in filenames:
				file_list.append(os.path.abspath(os.path.join(self.data_folder, f)))

		p=Pool()
		p.map(self.process_data, file_list[0])

	def process_data_multiprocess(self):
		if not os.path.exists(self.imaging_results_folder):
			os.makedirs(self.imaging_results_folder)

		self.calculate_parameters()
		processes=[]

		for dirpath,_,filenames in os.walk(self.data_folder):
			for f in filenames:
				p=multiprocessing.Process(target=self.process_data, args=(os.path.join(self.data_folder, f),))
				processes.append(p)
				p.start()

		for process in processes:
			process.join()

	#def process_data_seq(self):
	def process_data(self, file_location):
		s21=np.load(file_location)
		if self.win:
			s21=s21*np.hanning(s21.shape[1])
			s21=s21*np.hanning(s21.shape[0])[:,np.newaxis]

		I=np.zeros([self.npos], dtype = np.complex64)
		s21_arr=np.zeros([self.nx, self.nr], dtype=np.complex64)
		nc0=int(self.nfre/2.0) #first chunk of the frequency: f0,fc
		nc1=int((self.nfre+1)/2.0) #first chunk of the frequency: fc,ff
		s21_arr[:,0:nc1]=s21[:, nc0:self.nfre] #invert array order
		s21_arr[:,self.nr-nc0: self.nr]=s21[:, 0:nc0]
		Fn0=self.nr*ifft(s21_arr, n=self.nr)
		Fn=np.zeros([self.nx, self.nposy, self.nposx], dtype=np.complex64)

		"""
		for k in range(0,self.nx):
			Fn=np.interp(self.Rnk[k,:]-self.R0, self.rn, np.real(Fn0[k,:]))+1j*np.interp(self.Rnk[k,:]-self.R0, self.rn, np.imag(Fn0[k,:]))
			Fn*=np.exp(4j*np.pi*(self.fre_min/c0)*(self.Rnk[k,:]-self.R0))
			I+=Fn

		I/=(self.nfre*self.nx)
		I=np.reshape(I, (self.nposy, self.nposx))
		I=np.flipud(I)
		I.dump(os.path.join(self.imaging_results_folder, "image_{}".format(os.path.split(file_location)[1])))
		"""

	def process_data_fpfa(self, file_location):
		s21=np.load(file_location)
		if self.win:
			s21=s21*np.blackman(s21.shape[1])
			s21=s21*np.blackman(s21.shape[0])[:,np.newaxis]

		#img=np.fft.fft2(s21, s=(1000,1000,))
		img=np.fft.fft2(s21, s=(2000,2000))
		img=np.fft.fftshift(img, axes=0)
		#I=np.flipud(np.transpose(20*np.log10(np.absolute(img))))
		I=np.absolute(img)
		#I=np.flipud(np.transpose(I))
		I=np.transpose(I)
		I/=np.amax(I)
		I=(10*np.log10(I))
		#coordinate_transformation=

		img=I[620:1950,:]
		vmin, vmax=np.amin(img)+40, np.amax(img)

		#im=plt.imshow(I[400:800,40:120], cmap='jet', aspect='auto')
		im=plt.imshow(img, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
		#plt.imshow(self.masked_values, cmap='Greys', aspect='auto', extent=[self.xi,self.xf,self.yi,self.yf])
		plt.colorbar(im, orientation='vertical')
		plt.show()

	def _pickle_method(self,m):
		if m.im_self is None:
			return getattr, (m.im_class, m.im_func.func_name)
		else:
			return getattr, (m.im_self, m.im_func.func_name)

if __name__ == "__main__":
	xi=-350.0
	xf=350.0
	yi=100.0
	yf=950.0

	R0=0.0
	dx=0.5
	dy=0.5
	folder_name='data'

	x=jro_gbsar_processor(xi=xi, xf=xf, yi=yi, yf=yf, R0=R0, dx=dx, dy=dy, folder_name=folder_name, win=True)
	x.read_metadata()
	#x.process_data_pool()
	x.process_data_fpfa(file_location="/home/andre/Documents/andre/code/python/data/dset_20.dat")
