from __future__ import division
import fnmatch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import dates
from datetime import datetime
from scipy.fftpack import fft, ifft
from scipy import interpolate
from scipy import signal
import time
import h5py
import numpy.ma as ma
from datetime import datetime

RUTA_DATOS_PROCESADOS = "/home/andre/sar_processed_data/Imaging/"
FOLDER_DATOS_PROCESADOS = "datos_procesados_%s"
FECHA_HORA = "10.10.16_12.21.17"

RUTA = RUTA_DATOS_PROCESADOS + FOLDER_DATOS_PROCESADOS %FECHA_HORA + '/'

RUTA_DESPLAZAMIENTOS = "/home/andre/sar_processed_data/Sliding/"
NOMBRE_DESPLAZAMIENTOS = "Desplazamientos_%s/"

folders = []
c0  =  299792458.0
time_format = "%H.%M.%S"
date_format = "%d.%m.%y"

if not os.path.exists(RUTA_DESPLAZAMIENTOS + NOMBRE_DESPLAZAMIENTOS %FECHA_HORA):
	os.makedirs(RUTA_DESPLAZAMIENTOS + NOMBRE_DESPLAZAMIENTOS %FECHA_HORA)

for file in os.listdir(RUTA):
	if fnmatch.fnmatch(file, '*.hdf5'):
		folders.append(file)

std_values = np.zeros([len(folders) - 1], dtype = float)
mean_values = np.zeros([len(folders) - 1], dtype = float)
magnitude_mean_values = np.zeros([len(folders) - 1], dtype = float)
mean_sum_values = np.zeros([len(folders) - 1], dtype = float)
date_values = []
vicinity_length_x = 1.0
vicinity_length_y = 1.0

for x in range(1, len(folders)):
	for file in folders:
		datos = '%d.hdf5' %x
		if fnmatch.fnmatch(file, '*' + 'toma_' + datos):
			datos = file
			break

	file_temp = h5py.File(RUTA + datos, 'r')
	dset = file_temp["Complex_image"]
	dx = dset.attrs["dx"]
	dy = dset.attrs["dy"]
	date = dset.attrs["date"]
	t = dset.attrs["time"]
	Imagen_master = dset[...]
	file_temp.close()

	if x == 1:
		x_max = int(np.where(np.absolute(Imagen_master) == np.amax(np.absolute(Imagen_master)))[0])
		y_max = int(np.where(np.absolute(Imagen_master) == np.amax(np.absolute(Imagen_master)))[1])

	interest_region_master = np.zeros([int((2 * vicinity_length_x) / dx), int((2 * vicinity_length_y) / dy)], dtype = complex)

	interest_region_master = np.copy(Imagen_master[np.absolute(x_max - interest_region_master.shape[0]): x_max + interest_region_master.shape[0] - 1,
												  np.absolute(y_max - interest_region_master.shape[1]): y_max + interest_region_master.shape[1] - 1])
	for file in folders:
		datos = '%d.hdf5' %(x + 1)
		if fnmatch.fnmatch(file, '*' + 'toma_' + datos):
			datos = file
			break

	file_temp = h5py.File(RUTA + datos, 'r')
	dset = file_temp["Complex_image"]
	dx = dset.attrs["dx"]
	dy = dset.attrs["dy"]
	fi = dset.attrs["fi"]
	ff = dset.attrs["ff"]
	Imagen_slave = dset[...]
	file_temp.close()

	interest_region_slave = np.zeros([int((2 * vicinity_length_x) / dx), int((2 * vicinity_length_y) / dy)], dtype = complex)

	interest_region_slave = np.copy(Imagen_slave[np.absolute(x_max - interest_region_slave.shape[0]): x_max + interest_region_slave.shape[0] - 1,
												 np.absolute(y_max - interest_region_slave.shape[1]): y_max + interest_region_slave.shape[1] - 1])

	#complex_correlation = 10 * np.log10(interest_region_master * np.conj(interest_region_slave))
	complex_correlation = (interest_region_master * np.conj(interest_region_slave))

	magnitude_mean_values[x-1] = (np.mean(np.absolute(complex_correlation)))
	mean_values[x-1] = np.mean(np.angle(complex_correlation)) * 1000 * (c0/((fi + ff) * 4 * np.pi * 0.5))
	mean_sum_values[x-1] = np.sum(mean_values[0:x-1])
	std_values[x-1] = np.std(np.angle(complex_correlation)) * 1000 * (c0/((fi + ff) * 4 * np.pi * 0.5))
	date_values.append(datetime.strptime('.'.join((date, t)), '.'.join((date_format, time_format))))

magnitude_mean_values = magnitude_mean_values / np.amax(magnitude_mean_values)
fig = plt.figure(figsize = (15.0, 8.0))

plt.subplot(221)
plt.title('sliding mean vs time', fontsize = 12)
plt.plot_date(date_values, mean_values, ls = 'solid')
plt.ylabel('sliding mean (mm)', fontsize = 10)
plt.xlabel('time', fontsize = 10)

plt.subplot(222)
plt.title('sliding mean sum vs time', fontsize = 12)
plt.plot_date(date_values, mean_sum_values, ls = 'solid')
plt.ylabel('sliding mean sum (mm)', fontsize = 10)
plt.xlabel('time', fontsize = 10)

plt.subplot(223)
plt.title('sliding std. deviation vs time', fontsize = 12)
plt.plot_date(date_values, std_values, ls = 'solid')
plt.ylabel('sliding std. deviation (mm)', fontsize = 10)
plt.xlabel('time', fontsize = 10)

plt.subplot(224)
plt.title('magnitude mean vs time', fontsize = 12)
plt.plot_date(date_values, magnitude_mean_values, ls = 'solid')
plt.ylabel('magnitude mean', fontsize = 10)
plt.xlabel('time', fontsize = 10)

plt.tight_layout()
#plt.show()
plt.savefig(RUTA_DESPLAZAMIENTOS + NOMBRE_DESPLAZAMIENTOS %FECHA_HORA + "/mean_stddev_vicinity_%s.png" %(FECHA_HORA))
#plt.show()
