from __future__ import division
import fnmatch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import interpolate
from scipy import signal
import time
import h5py
import numpy.ma as ma
from datetime import datetime

RUTA_DATOS_PROCESADOS = "/home/andre/sar_processed_data/Imaging/"
FOLDER_DATOS_PROCESADOS = "datos_procesados_%s"
FECHA_HORA = "27.02.17_16.20.34"

folders = []
c0  =  299792458.0

RUTA = RUTA_DATOS_PROCESADOS + FOLDER_DATOS_PROCESADOS %FECHA_HORA + '/'

RUTA_DESPLAZAMIENTOS = "/home/andre/sar_processed_data/Sliding/"
NOMBRE_DESPLAZAMIENTOS = "Desplazamientos_%s/"

if not os.path.exists(RUTA_DESPLAZAMIENTOS + NOMBRE_DESPLAZAMIENTOS %FECHA_HORA):
	os.makedirs(RUTA_DESPLAZAMIENTOS + NOMBRE_DESPLAZAMIENTOS %FECHA_HORA)

for file in os.listdir(RUTA):
	if fnmatch.fnmatch(file, '*.hdf5'):
		folders.append(file)

#folders.remove('res_22.02.17_17.02.37_toma_1.hdf5')

for x in range(1, len(folders)):
#for x in range(2, len(folders)):
	for file in folders:
		datos = '%d.hdf5' %x
		if fnmatch.fnmatch(file, '*' + 'toma_' + datos):
			datos = file
			break

	f = h5py.File(RUTA + datos, 'r')
	dset = f["Complex_image"]
	Imagen_master = dset[...]
	if x == 1:
		xi = dset.attrs["xi"]
		xf = dset.attrs["xf"]
		yi = dset.attrs["yi"]
		yf = dset.attrs["yf"]
	f.close()

	for file in folders:
		datos = '%d.hdf5' %(x + 1)
		if fnmatch.fnmatch(file, '*' + 'toma_' + datos):
			datos = file
			break

	file_temp = h5py.File(RUTA + datos, 'r')
	dset = file_temp["Complex_image"]
	Imagen_slave = dset[...]
	file_temp.close()

	if x == 1:
		complex_correlation_num = (Imagen_master * np.conj(Imagen_slave))
		complex_correlation_den = np.sqrt(np.absolute(Imagen_master)**2 * np.absolute(Imagen_slave)**2)

	complex_correlation_num += (Imagen_master * np.conj(Imagen_slave))
	complex_correlation_den += np.sqrt(np.absolute(Imagen_master)**2 * np.absolute(Imagen_slave)**2)
	#complex_correlation = signal.convolve2d(complex_correlation, kernel)

complex_correlation = complex_correlation_num / complex_correlation_den

fig = plt.figure(1)
fig.suptitle("Complex correlation magnitude", fontsize = 14)
im = plt.imshow(np.absolute(complex_correlation), cmap = 'jet', aspect = 'auto', extent = [xi,xf,yi,yf])
cbar = plt.colorbar(im, orientation = 'vertical')
plt.ylabel('Range (m)', fontsize = 11)
plt.xlabel('Cross-range (m)', fontsize = 11)
plt.savefig(RUTA_DESPLAZAMIENTOS + NOMBRE_DESPLAZAMIENTOS %FECHA_HORA + "/complex_correlation_magnitude_%s.png" %(FECHA_HORA))
fig.clear()

threshold = 0.7
mask = np.absolute(complex_correlation)
low_value_indices = mask < threshold
high_value_indices = mask >= threshold
mask[low_value_indices] = 0
mask[high_value_indices] = 1

fig = plt.figure(1)
fig.suptitle("Mask", fontsize = 14)
im = plt.imshow(mask, cmap = 'Greys', interpolation = 'None', aspect = 'auto', extent = [xi,xf,yi,yf])
plt.ylabel('Range (m)', fontsize = 14)
plt.xlabel('Cross-range (m)', fontsize = 14)
plt.savefig(RUTA_DESPLAZAMIENTOS + NOMBRE_DESPLAZAMIENTOS %FECHA_HORA + "/mask_%s.png" %(FECHA_HORA))
fig.clear()

for x in range(1, len(folders)):
#for x in range(2, len(folders)):
	for file in folders:
		datos = '%d.hdf5' %x
		if fnmatch.fnmatch(file, '*' + 'toma_' + datos):
			datos = file
			break

	f = h5py.File(RUTA + datos, 'r')
	dset = f["Complex_image"]
	Imagen_master = dset[...]
	f.close()

	for file in folders:
		datos = '%d.hdf5' %(x + 1)
		if fnmatch.fnmatch(file, '*' + 'toma_' + datos):
			datos = file
			break

	file_temp = h5py.File(RUTA + datos, 'r')
	dset = file_temp["Complex_image"]
	Imagen_slave = dset[...]
	file_temp.close()

	complex_correlation_num = (Imagen_master * np.conj(Imagen_slave))
	complex_correlation_den = np.sqrt(np.absolute(Imagen_master)**2 * np.absolute(Imagen_slave)**2)

	aux = complex_correlation_num / complex_correlation_den

	fig = plt.figure(1)
	fig.suptitle("Sliding between image %d and %d" %(x, x+1))
	im = plt.imshow((c0/(16.05 * 1E9 * 4 * np.pi)) * np.angle(aux) * mask, cmap = 'jet', aspect = 'auto', extent = [xi,xf,yi,yf])
	cbar = plt.colorbar(im, orientation = 'vertical')
	plt.ylabel('Range (m)', fontsize = 14)
	plt.xlabel('Cross-range (m)', fontsize = 14)
	plt.savefig(RUTA_DESPLAZAMIENTOS + NOMBRE_DESPLAZAMIENTOS %FECHA_HORA + "/Desp_%s_toma_%d_%d.png" %(FECHA_HORA, x, (x+1)))
	fig.clear()
