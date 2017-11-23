import numpy as np
import h5py
import time
import os, sys
import threading

from constants import *
from parameters import *
from datetime import datetime

sys.path.append('/home/Documents/sar-app')

from api import vna, rail
from common.db import *
from config import *
from static import *

class sar_experiment(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

        query = configuration_collection.find_one({"name" : "current_configuration"})
        current_configuration_id = query['config_id']
        current_configuration = configuration_collection.find_one({"_id" : current_configuration_id})

        self.collection_name = current_configuration['collection_name']
        self.xi = current_configuration['xi']
        self.xf = current_configuration['xf']
        self.fi = current_configuration['fi']
        self.ff = current_configuration['ff']
        self.nfre = current_configuration['nfre']
        self.ptx = current_configuration['ptx']
        self.ifbw = current_configuration['ifbw']
        self.beam_angle = current_configuration['beam_angle']

        self.beam_angle = (180 - beam_angle) / 2.0
        self.xi = int(self.xi * METERS_TO_STEPS_FACTOR)
        self.xf = int(self.xf * METERS_TO_STEPS_FACTOR)
        self.dx = int((c0  / (4.0 * self.ff * 1E9 * np.cos(self.beam_angle * np.pi / 180.0))) * METERS_TO_STEPS_FACTOR)
        self.npos = int(((self.xf - self.xi) / self.dx) + 1.0)
        self.xf = self.xi + self.dx * (self.npos - 1)
        self.stop_flag = False

    def run(self):
        vector_network_analyzer = vna.vnaClient()
        vector_network_analyzer.connect()
        vector_network_analyzer.send_ifbw(self.ifbw)
        vector_network_analyzer.send_number_points(self.nfre)
        vector_network_analyzer.send_freq_start(freq_start = self.fi)
        vector_network_analyzer.send_freq_stop(freq_stop = self.ff)
        vector_network_analyzer.send_power(self.ptx)
        vector_network_analyzer.send_select_instrument()
        vector_network_analyzer.send_cfg()
        time.sleep(5)

        rail = rail.railClient()
        rail.connect()
        rail.zero()

        query=experiment_collection.find_one({"name" : "current_experiment"})

        if query is None:
            start_time = datetime.utcnow().replace(tzinfo = FROM_ZONE)
            start_time = take_time.astimezone(TO_ZONE)
            current_experiment = "{}_{}".format(self.collection_name, strftime("%d-%m-%y_%H:%M:%S"))
            experiment_path = DATA_PATH + current_experiment

            os.mkdir(experiment_path)
            #should log the folder creation
            data_take = 1

            post = {"name" : "current_experiment",
                    "folder_name" : current_experiment,
                    "status" : "running",
                    "last_data_take" : data_take}
            experiment_collection.insert_one(post)

        status = query['status']

        if status is not None:
            if status=='running':
                data_take=query['last_data_take']
                folder_name=query['folder_name']

            if status=='not running':
                start_time = datetime.utcnow().replace(tzinfo = FROM_ZONE)
                start_time = start_time.astimezone(TO_ZONE)
                experiment_path = DATA_PATH + self.collection_name

                os.mkdir(experiment_path)
                data_take = 1

                current_experiment = "{}_{}".format(self.collection_name, strftime("%d-%m-%y_%H:%M:%S"))
                post = {"name" : "current_experiment",
                        "folder_name" : current_experiment,
                        "status" : "running",
                        "last_data_take" : data_take}
                experiment_collection.find_one_and_update({'_id': query['_id']}, '$set': post)

        while True:
            if not self.xi == 0:
                rail.send_move(self.xi, 'R')
                time.sleep(2)

            file_name = 'dset_{}.hdf5'.format(data_take)
            f = h5py.File(experiment_path + file_name, 'w')
            dset = f.create_dataset('sar_dataset', (self.npos, self.nfre), dtype = np.complex64)
            data = vector_network_analyzer.send_sweep()
            dset[0,:] = data

            for j in range(1, self.npos):
                if self.stop_flag:
                    rail.close()
                    vector_network_analyzer.close()
                    f.close()
                    os.remove(experiment_path + file_name)
                    break

                rail.send_move(self.dx, 'R')
                time.sleep(2)
                data = vector_network_analyzer.send_sweep()
                dset[j,:] = data

            take_time = datetime.utcnow().replace(tzinfo = FROM_ZONE)
            take_time = take_time.astimezone(TO_ZONE)

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

            if self.stop_flag:
                f.close()
                vector_network_analyzer.close()
                rail.close()
                break

    	    f.close()
            rail.send_zero_position()
            g.open(APP_PATH + "/tmp/last_data_take.tmp", "w")
            g.write(str(data_take))
            data_take = data_take + 1

    def stop(self):
        self.stop_flag = True

if __name__ == '__main__':
    pass
