from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import ifft, fft, ifft2, fftshift
import fnmatch
import time
import h5py
import os.path as path
from scipy.interpolate import griddata, interp1d

np.seterr(divide='ignore', invalid='ignore')

DATE_TIME = '10.10.16_12.21.17'

#FOLDER_DATA = '/home/andre/sar_raw_data/experimentos_%s/'
FOLDER_DATA = 'test-data/experimentos_%s/'
NAME_DATA = 'datos_toma_%d.hdf5'
folders = []

for file in os.listdir(FOLDER_DATA %DATE_TIME):
	if fnmatch.fnmatch(file, '*.hdf5'):
		folders.append(file)

for n_toma in range(1,2):
	tic = time.time()
	print "Procesando datos %s toma %d..." %(DATE_TIME, n_toma)
	f = h5py.File(FOLDER_DATA %DATE_TIME + NAME_DATA %(n_toma), 'r')
	dset = f['sar_dataset']

	dax = dset.attrs['dx']
	nx = dset.attrs['npos']
	fre_min = dset.attrs['fi']
	fre_max = dset.attrs['ff']
	nfre = dset.attrs['nfre']
	s21 = dset[...]
	f.close()

	'''
	data parameters
	'''
	c0  =  299792458.0
	B = fre_max - fre_min
	fre_cen = (fre_max + fre_min) / 2

	'''
	image paramters
	'''
	#image size

	r0 = 0.0
	rf = 15.0
	cr0 = -3.0
	crf = 3.0

	#matched filter center distance X
	#rs = (r0 + rf) / 2
	rs = 2.0

	#cross range resolution
	cross_range_res = 2 ** 12
	range_res = 2 ** 10

	#cross range zero padding to expand limits on that direction
	rows = int((max(cross_range_res, len(s21)) - len(s21)) / 2)
	cols = int((max(range_res, s21.shape[1]) - s21.shape[1]) / 2)
	cols_left = cols
	cols_right = cols
	rows_up = rows
	rows_down = rows

	if not (len(s21) % 2) == 0:
		rows_down = rows + 1

	if not (s21.shape[1] % 2) == 0:
		cols_left = cols+1

	s21 = np.pad(s21,[[rows_up,rows_down],[cols_left, cols_right]], 'constant', constant_values=0)

	#w-k space transformation
	k = 4 * np.pi / c0 * np.linspace((fre_cen - B/2) / 4, (fre_cen + B/2)/ 4 , s21.shape[1])
	ku = np.pi / dax * np.linspace(-1, 1, s21.shape[0]) * (1/2**4)
	kk, kuu = np.meshgrid(k, ku)
	kxmn = np.sqrt(kk**2 - kuu**2)
	kxmn = np.nan_to_num(kxmn)
	kymn = kuu
	kymin = np.amin(kymn)
	kymax = np.amax(kymn)
	kxmin = np.amin(kxmn)
	kxmax = np.amax(kxmn)

	#matched filter s0
	s0 = np.exp(1j * rs * (kxmn - kk))
	Fmn = np.fft.fftshift(fft(s21, axis = 0))
	Fmn *= s0
	ky_len = len(kxmn)

	#make kx an evenly spaced axis

	res = 2**12
	kx_even = np.linspace(kxmin , kxmax , res)
	Fst = np.zeros((ky_len, res), dtype = np.complex64)

	for i in range(ky_len):
		interp_fn = interp1d(kxmn[i], Fmn[i], bounds_error = False, fill_value=(0,0))
		Fst[i] = interp_fn(kx_even)

	Fst = np.nan_to_num(Fst)

	f = ifft2(Fst)
	f = (np.rot90(f))
	f = np.flipud(f)

	max_range = 2 * np.pi * f.shape[0]/(kxmax - kxmin)
	max_crange = 2 * np.pi * f.shape[1]/(kymax - kymin)

	print max_range, max_crange

	r_index = [int(round((r0/max_range)*f.shape[0])),
			   int(round((rf/max_range)*f.shape[0]))]
	cr_index = [int(round(f.shape[1]*((cr0+ky_len*dax/2.0)/(ky_len*dax)))),
				int(round(f.shape[1]*((crf+ky_len*dax/2.0)/(ky_len*dax))))]

	trunc_image = f[r_index[0]:r_index[1],cr_index[0]:cr_index[1]]
	trunc_image = np.flipud(trunc_image)
	#trunc_image = abs(20 * np.log10(trunc_image))
	#trunc_image = abs(trunc_image)
	trunc_image = np.flipud(abs(f))

	extent = [cr0, crf, r0, rf]

	#im = plt.imshow(trunc_image, cmap = 'jet', aspect = 'auto', extent=extent)
	im = plt.imshow(trunc_image, cmap = 'jet', aspect = 'auto')
	cbar = plt.colorbar(im, orientation = 'vertical')

	print 'Demoro %.4f segundos' %(time.time() - tic)

	plt.show()
