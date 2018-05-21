from __future__ import division
import os, time, h5py
import numpy as np
import matplotlib.pyplot as plt
import mwavepy as mv
import re

from scipy.fftpack import fft, ifft
from scipy import interpolate

daniel_data_path='/home/andre/Documents/sar/tesis/tesis-daniel/data'
c0 = 299792458.0
R0=1.85

new_list=[]

for (dirpath, dirnames, filenames) in os.walk(daniel_data_path):
    for index, element in enumerate(filenames):
        m=re.match("5.(\\d+).s1p", element)
        if not m:
            continue
        pair=[int(m.group(1)),m.group(0)]
        new_list.append(pair)
        pair=[]

for (dirpath, dirnames, filenames) in os.walk(daniel_data_path):
    for element in sorted(new_list):
        data=mv.Network(os.path.join(dirpath, element[1]))
        if element[0]==0:
            freq=data.frequency.f
            B=freq[-1]-freq[0]
            nfre=len(freq)

            dr=c0/(2*B)
            df=(freq[-1]-freq[0])/(nfre-1)

            fact=1
            #nr=2**int(np.ceil(np.log2(nfre*fact)))
            nr=nfre

            n=np.arange(nr)
            B=df*nr
            dr=c0/(2*B)
            distance=np.arange(nr)*dr
            nc0=int(nfre/2.0)
            nc1=int((nfre+1)/2.0)
            freq=np.arange(nfre)
            min_value=0
            max_value=0
            rows=len(new_list)
            s21_arr=np.zeros((rows, nr), dtype=complex)

        s21=data.s.flatten()

        #s21*=np.exp((4j*np.pi*freq*df*R0)/c0)
        s21_arr[element[0],0:nc1]=s21[nc0:nfre]
        s21_arr[element[0],nr-nc0:nr] = s21[0:nc0]

        s21_arr[element[0],:]=ifft(s21_arr[element[0],:])
        #s21_arr[element[0],:]=np.absolute(s21_arr[element[0],:])
        #print s21_arr[element[0]]

        if element[0]==0:
            min_value=np.amin(np.absolute(s21_arr[element[0],:]))
            max_value=np.amax(np.absolute(s21_arr[element[0],:]))
            continue
        else:
            if np.amin(np.absolute(s21_arr[element[0],:]))<min_value:
                min_value=np.amin(np.absolute(s21_arr[element[0],:]))
            if np.amax(np.absolute(s21_arr[element[0],:]))>max_value:
                max_value=np.amax(np.absolute(s21_arr[element[0],:]))


fig=plt.figure(1)
plt.xlabel('Range (m)', fontsize = 14)
plt.ylabel('Relative Radar Reflectivity (dB)', fontsize = 14)

for row in range(rows):
    ax=plt.gca()
    ax.set_ylim([-1.1*np.pi, 1.1*np.pi])
    #ax.set_xlim([0, nfre*dr])
    plt.xlabel('Range (m)', fontsize = 14)
    #plt.ylabel('Relative Radar Reflectivity (dB)', fontsize = 14)
    plt.ylabel('Phase (deg)', fontsize = 14)
    #plt.plot(distance, np.absolute(s21_arr[row,:]))
    plt.plot(distance[70:105], np.angle(s21_arr[row,70:105]))
    print "dist: ",distance[85:89]
    print "fase: ",np.angle(s21_arr[row,85:89])
    plt.savefig(os.path.join(dirpath, '{}.png'.format("{0:0{width}}".format(row, width=rows))))
    fig.clear()

'''
data=mv.Network(daniel_data_path)
s21=data.s.flatten()
freq=data.frequency.f
B=freq[-1]-freq[0]
nfre=len(freq)

c0 = 299792458.0
dr = c0/(2*B)
df = (freq[-1]-freq[0])/(nfre-1)
R0=1.85

fact=1
nr=2 ** int(np.ceil(np.log2(nfre*fact)))

n  = np.arange(nr)
B  = df*nr
dr = c0 / (2*B)

win = False

distance=np.arange(nr) * dr

nc0=int(nfre/2.0)
nc1=int((nfre+1)/2.0)

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
s21_arr = np.zeros(nr, dtype = complex)

s21*=np.exp((4j*np.pi*freq*df*R0)/c0)

s21_arr[0:nc1]=s21[nc0:nfre]
s21_arr[nr-nc0:nr] = s21[0:nc0]

s21_arr = ifft(s21_arr)
s21_arr = np.absolute(s21_arr)

fig = plt.figure(1)
im = plt.plot(distance, s21_arr)
plt.xlabel('Range (m)', fontsize = 14)
plt.ylabel('Relative Radar Reflectivity (magnitude)', fontsize = 14)
plt.savefig('4bunker9.png')


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
'''
