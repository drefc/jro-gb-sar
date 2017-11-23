from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy import interpolate
import time
import h5py

FECHA_HORA = '19.08.2016_16.12.10'
RUTA = '/home/cielosar/Desktop/Datos_hdf5/experimentos_%s/'
NOMBRE_ARCHIVO = 'datos_toma_%d.hdf5'

LONG_VUELTA = 66
PASOS_VUELTA = 20000
FACTOR_LONG_PASOS = 1000 * PASOS_VUELTA / LONG_VUELTA

for n_toma in range(1,2):
    f = h5py.File(RUTA % FECHA_HORA + NOMBRE_ARCHIVO %(n_toma), 'r')
    dset = f['sar_dataset']
    fre_min = dset.attrs['fi']
    fre_max = dset.attrs['ff']
    nx = dset.attrs['npos']
    nfre = dset.attrs['nfre']
    npos = dset.attrs['npos']
    s21 = np.empty([nx, nfre], dtype = np.complex64)
    s21 = dset[...]
    f.close()
    R0 = 0

    c0 = 299792458.0
    B = fre_max - fre_min
    dr = c0 / (2 * B)
    df = (fre_max - fre_min) / (nfre - 1)

    fact  =  1
    nr = 2 ** int(np.ceil(np.log2(nfre*fact)))

    n  = np.arange(nr)
    B  = df*nr
    dr = c0 / (2*B)

    win = False

    distance = np.arange(nr) * dr

    nc0 = int(nfre/2.0)
    nc1 = int((nfre+1)/2.0)

    if win:
		#Window Range
		Wr = 1.0 - np.cos(2*np.pi*np.arange(1,nfre+1)/(nfre+1))
		for k in range(0,nx):
			s21[k,:] = Wr * s21[k,:]
		#Window Cross-range
		Wx = 1.0 - np.cos(2*np.pi*np.arange(1,nx+1)/(nx+1))
		for f in range(0,nfre):
			s21[:,f] = Wx * s21[:,f]


    freq = np.arange(nfre)
    s21_arr = np.zeros([npos, nr], dtype = complex)

    s21 *= np.exp((4j * np.pi * R0 * freq  * df)/ c0)

    s21_arr[:,0:nc1] = s21[:,nc0:nfre]
    s21_arr[:,nr-nc0:nr] = s21[:,0:nc0]

    s21_arr = ifft(s21_arr)
    s21_arr = np.absolute(s21_arr)

    fig = plt.figure(1)
    im = plt.plot(distance, s21_arr[100,:])
    plt.xlabel('Range (m)', fontsize = 14)
    plt.ylabel('Relative Radar Reflectivity (dB)', fontsize = 14)

    plt.show()
