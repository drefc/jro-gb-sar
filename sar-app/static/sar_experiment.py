import numpy as np
import h5py
import time
import os, sys
import threading

from constants import *
from parameters import *
from datetime import datetime
from tasks import send_data

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from api import vna, rail
from common.db import *
from config import *
from static import *

class sar_experiment(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        query=configuration_collection.find_one({"_id":"current_configuration"})
        current_configuration=query['configuration']

        '''
        The config shall be checked by the web interface before starting an experiment
        '''

        self.collection_name=current_configuration['collection_name']
        self.xi=current_configuration['xi']
        self.xf=current_configuration['xf']
        self.fi=current_configuration['fi']
        self.ff=current_configuration['ff']
        self.nfre=current_configuration['nfre']
        self.ptx=current_configuration['ptx']
        self.ifbw=current_configuration['ifbw']
        self.beam_angle=current_configuration['beam_angle']

        self.beam_angle = (180 - beam_angle) / 2.0
        self.xi = int(self.xi * METERS_TO_STEPS_FACTOR)
        self.xf = int(self.xf * METERS_TO_STEPS_FACTOR)
        self.dx = int((c0  / (4.0 * self.ff * 1E9 * np.cos(self.beam_angle * np.pi / 180.0))) * METERS_TO_STEPS_FACTOR)
        self.npos = int(((self.xf - self.xi) / self.dx) + 1.0)
        self.xf = self.xi + self.dx * (self.npos - 1)
        self.stop_flag = False

    def run(self):
        vector_network_analyzer=vna.vnaClient()
        vector_network_analyzer.connect()
        vector_network_analyzer.send_ifbw(self.ifbw)
        vector_network_analyzer.send_number_points(self.nfre)
        vector_network_analyzer.send_freq_start(freq_start = self.fi)
        vector_network_analyzer.send_freq_stop(freq_stop = self.ff)
        vector_network_analyzer.send_power(self.ptx)
        vector_network_analyzer.send_select_instrument()
        vector_network_analyzer.send_cfg()

        rail=rail.railClient()
        rail.connect()
        rail.zero()

        query=experiment_collection.find_one({"_id":"current_experiment"})

        if query:
            data_take=query['experiment']['last_data_take']
            folder_name=query['experiment']['folder_name']
        else:
            start_time=datetime.utcnow().replace(tzinfo=FROM_ZONE)
            start_time=take_time.astimezone(TO_ZONE)
            folder_name="{}_{}".format(self.collection_name, start_time.strftime("%d-%m-%y_%H:%M:%S"))
            experiment_path=DATA_PATH+folder_name

            os.mkdir(experiment_path)
            #should log the folder creation
            data_take = 1

            experiment={"folder_name":folder_name,
                        "last_data_take":data_take}
            experiment_collection.find_one_and_update({"_id":"current_experiment"},
                                                      {"$set":{"experiment": experiment}},
                                                      upsert=True)

        while True:
            if self.xi!=0:
                rail.move(self.xi, 'R')

            file_name='dset_{}.hdf5'.format(data_take)
            file_path=experiment_path+file_name
            f=h5py.File(file_path, 'w')
            dset=f.create_dataset('sar_dataset', (self.npos, self.nfre), dtype = np.complex64)
            dset.attrs['xi'] = self.xi / METERS_TO_STEPS_FACTOR
            dset.attrs['xf'] = self.xf / METERS_TO_STEPS_FACTOR
            dset.attrs['dx'] = self.dx / METERS_TO_STEPS_FACTOR
            dset.attrs['npos'] = self.npos
            dset.attrs['fi'] = self.fi * 1E9
            dset.attrs['ff'] = self.ff * 1E9
            dset.attrs['nfre'] = self.nfre
            dset.attrs['ptx'] = self.ptx
            dset.attrs['ifbw'] = self.ifbw
            dset.attrs['beam_angle'] = self.beam_angle
            dset.attrs['datetime'] = take_time.strftime("%d-%m-%y %H:%M:%S")

            data=vector_network_analyzer.send_sweep()
            dset[0,:]=data

            for j in range(1, self.npos):
                if self.stop_flag:
                    rail.close()
                    vector_network_analyzer.close()
                    f.close()
                    break

                rail.send_move(self.dx, 'R')
                data = vector_network_analyzer.send_sweep()
                dset[j,:] = data

            take_time = datetime.utcnow().replace(tzinfo = FROM_ZONE)
            take_time = take_time.astimezone(TO_ZONE)

            if self.stop_flag:
                f.close()
                vector_network_analyzer.close()
                rail.close()
                break

    	    f.close()
            send_data(file_path)
            rail.send_zero_position()
            data_take=data_take+1

        if self.stop_flag:
            os.remove(file_path)

    def stop(self):
        self.stop_flag = True
