from __future__ import division
import fnmatch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, HourLocator, DateFormatter
from datetime import datetime
import time
import h5py
import numpy.ma as ma
from datetime import datetime

RUTA_DATOS_PROCESADOS = "/home/andre/sar_processed_data/Imaging/"
FOLDER_DATOS_PROCESADOS = "datos_procesados_%s"

EXP_NAME = '17d05m2017y15h41m18s'
#RUTA = '/home/andre/Desktop/db_resultados/'
RUTA = '/home/andre/sar_processed_data/imaging/%s/data/' %EXP_NAME


RUTA_RESULTADOS = '/home/andre/sar_processed_data/sliding/%s/' %EXP_NAME

if not os.path.exists(RUTA_RESULTADOS):
	os.makedirs(RUTA_RESULTADOS)

folders = []
c0  =  299792458.0
threshold = 0.9
time_format = "%H:%M:%S"
date_format = "%Y-%m-%d"

for file in os.listdir(RUTA):
	if fnmatch.fnmatch(file, '*.hdf5'):
		folders.append(file)

phase_std_dev = np.zeros([len(folders)], dtype = float)
phase_mean = np.zeros([len(folders)], dtype = float)
mag_mean = np.zeros([len(folders) - 1], dtype = float)
phase_mean_sum = np.zeros([len(folders) - 1], dtype = float)
date = np.zeros([len(folders)], dtype = float)

for x in range(len(folders)-1):
	for file in folders:
		datos = 'image%d.hdf5' %x

		if fnmatch.fnmatch(file,  datos):
			datos = file
			break

	file_temp = h5py.File(RUTA + datos, 'r')
	dset = file_temp["Complex_image"]
	if x == 0:
		xi = dset.attrs["xi"]
		xf = dset.attrs["xf"]
		yi = dset.attrs["yi"]
		yf = dset.attrs["yf"]
		fi = dset.attrs["fi"]
		ff = dset.attrs["ff"]
	Imagen_master = dset[...]
	file_temp.close()

	for file in folders:
		datos = 'image%d.hdf5' %(x + 1)

		if fnmatch.fnmatch(file, datos):
			datos = file
			break

	file_temp = h5py.File(RUTA + datos, 'r')
	dset = file_temp["Complex_image"]
	Imagen_slave = dset[...]
	file_temp.close()

	if x == 0:
		complex_correlation_num = (Imagen_master * np.conj(Imagen_slave))
		complex_correlation_den = np.sqrt(np.absolute(Imagen_master)**2 * np.absolute(Imagen_slave)**2)
		continue

	complex_correlation_num += (Imagen_master * np.conj(Imagen_slave))
	complex_correlation_den += np.sqrt(np.absolute(Imagen_master)**2 * np.absolute(Imagen_slave)**2)

#this is the complex_correlation between the set of images
complex_correlation = complex_correlation_num / complex_correlation_den

#make plot mask for beamwidth
nposy = complex_correlation.shape[0]
nposx = complex_correlation.shape[1]

plot_mask = np.zeros((nposy, nposx))
count = 0

for k in range(nposy):
	if k >= (nposy - int(nposx / 2) + 1):
		plot_mask[k, 0:count] = 1
		plot_mask[k, nposx - count -1:nposx-1] = 1
		count = count + 1

plot_masked_values = np.ma.masked_where(plot_mask == 0, plot_mask)

#store the values
fig = plt.figure(1)
fig.suptitle("Complex correlation magnitude", fontsize = 14)
im = plt.imshow(np.absolute(complex_correlation), cmap = 'jet', aspect = 'auto',
				extent = [xi,xf,yi,yf])
plt.imshow(plot_masked_values, cmap = 'Greys', aspect = 'auto', extent = [xi,xf,yi,yf])
cbar = plt.colorbar(im, orientation = 'vertical')
plt.ylabel('Range (m)', fontsize = 10)
plt.xlabel('Cross-range (m)', fontsize = 10)
#plt.savefig(RUTA_DESPLAZAMIENTOS + NOMBRE_DESPLAZAMIENTOS %FECHA_HORA + "complex_correlation_magnitude_%s.png" %FECHA_HORA)
plt.savefig(RUTA_RESULTADOS + "complex_correlation_mag.png")
fig.clear()

mask = np.absolute(complex_correlation)
low_value_index = mask < threshold
high_value_index = mask >= threshold
mask[low_value_index] = 0
mask[high_value_index] = 1
masked = np.ma.masked_where(mask == 0, mask)

fig = plt.figure(1)
fig.suptitle("mask", fontsize = 14)
im = plt.imshow(mask, cmap = 'Greys', interpolation = 'None',
				aspect = 'auto', extent = [xi,xf,yi,yf])
plt.imshow(plot_masked_values, cmap = 'Greys', aspect = 'auto', extent = [xi,xf,yi,yf])
plt.ylabel('Range (m)', fontsize = 10)
plt.xlabel('Cross-range (m)', fontsize = 10)
plt.savefig(RUTA_RESULTADOS + "mask.png")
fig.clear()

date_values = []
#after calculating complex correlation, mask out the unwanted values and calculate
#the mean and standard deviation
#for x in range(1, len(folders)):
lambda_d = 1000 * (c0/((fi + ff) * 4 * np.pi * 0.5))

for x in range(len(folders)-1):
	print "Processig %d out of %d" %(x + 1, len(folders))
	for file in folders:
		datos = 'image%d.hdf5' %x

		if fnmatch.fnmatch(file,  datos):
			datos = file
			break

	file_temp = h5py.File(RUTA + datos, 'r')
	dset = file_temp["Complex_image"]
	Imagen_master = dset[...]
	file_temp.close()

	for file in folders:
		datos = 'image%d.hdf5' %(x + 1)

		if fnmatch.fnmatch(file, datos):
			datos = file
			break

	file_temp = h5py.File(RUTA + datos, 'r')
	dset = file_temp["Complex_image"]
	time = dset.attrs["time"]
	date = dset.attrs["date"]

	Imagen_slave = dset[...]
	file_temp.close()

	phase = np.angle(Imagen_master * np.conj(Imagen_slave))
	magnitude = np.absolute(Imagen_master)
	#masked_values_angle = ma.array(phase, mask = mask)
	masked_values_angle = np.ma.masked_where(mask == 0, phase)
	#masked_values_angle = np.flipud(masked_values_angle)
	masked_values_magnitude = ma.array(magnitude, mask = mask)

	fig = plt.figure(1)
	fig.suptitle("Image %d and %d" %(x, x+1))
	plt.ylabel('Range (m)', fontsize = 14)
	plt.xlabel('Cross-range (m)', fontsize = 14)

	if x == 0:
		vmin = -np.pi * lambda_d
		vmax = np.pi * lambda_d
		im = plt.imshow(lambda_d * np.angle(Imagen_master * np.conj(Imagen_slave)),
						cmap = 'jet', aspect = 'auto', extent = [xi,xf,yi,yf], vmin = vmin, vmax = vmax)
		plt.imshow(masked, 'gray', interpolation = 'none', aspect = 'auto', extent = [xi,xf,yi,yf])
		plt.imshow(plot_masked_values, cmap = 'Greys', aspect = 'auto', extent = [xi,xf,yi,yf])
		cbar = plt.colorbar(im, orientation = 'vertical', format='%.2f')

	im = plt.imshow(lambda_d * np.angle(Imagen_master * np.conj(Imagen_slave)),
					cmap = 'jet', aspect = 'auto', extent = [xi,xf,yi,yf], vmin = vmin, vmax = vmax)
	plt.imshow(masked, 'gray', interpolation = 'none', aspect = 'auto', extent = [xi,xf,yi,yf])
	plt.imshow(plot_masked_values, cmap = 'Greys', aspect = 'auto', extent = [xi,xf,yi,yf])

	plt.savefig(RUTA_RESULTADOS + "toma_%d_%d.png" %(x, (x+1)))

	mag_mean[x-1] = masked_values_magnitude.mean()
	phase_mean[x-1] = masked_values_angle.mean() * lambda_d
	phase_std_dev[x-1] = np.std(masked_values_angle) * lambda_d
	date_values.append(datetime.strptime(''.join((date, time)), ''.join((date_format, time_format))))
	phase_mean_sum[x-1] = np.sum(phase_mean)

phase_mean = phase_mean[:len(phase_mean) - 1]
phase_std_dev = phase_std_dev[:len(phase_std_dev) - 1]

fig.clear()
fig = plt.figure(figsize = (15.0, 8.0))

plt.subplot(221)
plt.title(r'$\overline{\Delta r}\/\/vs\/\/time$', fontsize = 16)
plt.ylabel(r'$\overline{\Delta r}\/\/(mm)$', fontsize = 16)
plt.xlabel(r'$time$', fontsize = 16)
plt.plot(date_values, phase_mean)
ax = plt.gca()
ax.set_ylim([-(np.amax(phase_mean) * 1.2), (np.amax(phase_mean) * 1.2)])
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
plt.setp(labels, rotation=30, fontsize=10)

plt.tight_layout()
plt.savefig(RUTA_RESULTADOS + "statistical_report.png")
