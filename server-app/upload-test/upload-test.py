import requests, h5py
import time, os
import numpy as np

from datetime import datetime

url_upload="http://25.56.104.172:4500/data_upload"
tmp_folder="/tmp/data_test_folder"
test_len=20

collection_name="test-collection"
start_position=0.0
stop_position=1.4
npos=150
delta=0.09
start_freq=15.5
stop_freq=15.6
nfre=1001
beam_angle=75.0
ptx='HIGH'

for index in range(test_len):
    file_name="dset_{}.h5py".format(index+1)
    file_location=os.path.join(tmp_folder, file_name)

    print "trying to upload file #{}".format(index+1)

    tmp=h5py.File(file_location, 'w')
    dset=tmp.create_dataset('sar_dataset', (100,100), data=np.random.rand(100, 100))
    dset.attrs['take_index']=index+1
    dset.attrs['xi']=start_position
    dset.attrs['xf']=stop_position
    dset.attrs['npos']=npos
    dset.attrs['dx']=delta
    dset.attrs['fi']=start_freq
    dset.attrs['ff']=stop_freq
    dset.attrs['nfre']=nfre
    dset.attrs['beam_angle']=beam_angle
    dset.attrs['ptx']=ptx
    take_time=datetime.utcnow()
    dset.attrs['datetime']=take_time.strftime("%d-%m-%y %H:%M:%S")
    tmp.close()

    files={'file': (file_name, open(file_location, 'rb'))}
    headers={'collection_name': collection_name}
    r=requests.post(url_upload, files=files, headers=headers)
    print r.text
    time.sleep(1 * 60)
