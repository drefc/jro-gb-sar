import numpy as np
import h5py
import time
import os, sys
import threading
import atexit
import RPi.GPIO as GPIO

from constants import *
from parameters import *
from datetime import datetime
from proj.tasks import send_data

print os.path.abspath(os.path.join(os.getcwd(), os.pardir))

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.getcwd())

from api import vna, rail
from common.db import *
from config import *
from static import *

class sar_experiment(threading.Thread):
    self.reset_arduino()

    def __init__(self):
	#thread constructor
        threading.Thread.__init__(self)	
	#creates rail and vna objects to be available throughout the instance
	self.rail=rail.railClient()
	self.vna=vna.vnaClient()
	#cleanup before program termination
	atexit.register(self.cleanup)
	
	#find the current configuration set by the user
        query=configuration_collection.find_one({"_id":"current_configuration"})
        current_configuration=query['configuration']

	self.collection_name=current_configuration['collection_name']
        self.xi=current_configuration['xi']
        self.xf=current_configuration['xf']
        self.fi=current_configuration['fi']
        self.ff=current_configuration['ff']
        self.nfre=current_configuration['nfre']
        self.ptx=current_configuration['ptx']
        self.ifbw=current_configuration['ifbw']
        self.beam_angle=current_configuration['beam_angle']
	
	#calculate the 'dx' and 'npos' parameters
	#recalculate 'xf'
        self.xi = int(self.xi * METERS_TO_STEPS_FACTOR)
        self.xf = int(self.xf * METERS_TO_STEPS_FACTOR)
        self.dx = int((c0  / (4.0 * self.ff * 1E9 * np.cos(0.5 * (180.0 - self.beam_angle) * np.pi / 180.0))) * METERS_TO_STEPS_FACTOR)
        self.npos = int(((self.xf - self.xi) / self.dx) + 1.0)
        self.xf = self.xi + self.dx * (self.npos - 1)
        self.stop_flag = False

    def run(self):
	#connect to the vna
        self.vna.connect()
        self.vna.send_ifbw(self.ifbw)
        self.vna.send_number_points(self.nfre)
        self.vna.send_freq_start(freq_start = self.fi)
        self.vna.send_freq_stop(freq_stop = self.ff)
        self.vna.send_power(self.ptx)
        self.vna.send_select_instrument()
        self.vna.send_cfg()
	#connect to the rail
	self.rail.connect()
	self.rail.zero()
	#find if there was an existing experiment running
        query=experiment_collection.find_one({"_id":"current_experiment"})
	#if there was, search for the 'last_data_take' and the 'folder_name' variables
	#last_data_take: stores the last data take that was FULLY COMPLETED
	#folder_name: folder where the data from the current experiment is being stored
        if query:
            data_take=query['experiment']['last_data_take']
            folder_name=query['experiment']['folder_name']
	    experiment_path=DATA_PATH+folder_name
        else:
	#if there was not, create a new folder, and append the current datetime to avoid conflicts with repeated 'collection_name' parameters
	#update (or create if not exists) the 'current_experiment' collection
            start_time=datetime.utcnow().replace(tzinfo=FROM_ZONE)
            start_time=start_time.astimezone(TO_ZONE)
            folder_name="{}_{}/".format(self.collection_name, start_time.strftime("%d-%m-%y_%H:%M:%S"))
            experiment_path=DATA_PATH+folder_name

            os.mkdir(experiment_path)
            #PENDING: should log the folder creation
            data_take = 1

            experiment={"folder_name":folder_name,
                        "last_data_take":data_take}
            experiment_collection.find_one_and_update({"_id":"current_experiment"},
                                                      {"$set":{"experiment": experiment}},
                                                      upsert=True)
	#the experiment loop begins
	#this will loop infinitely until an 'stop' is requested, or there is a power failure and the system needs to go down
        while True:
	    #move to the starting position (only if it is different from 0)
            if self.xi!=0:
                self.rail.move(self.xi, 'R')	
	    #the datasets will be named 'dset_{data_take}.hdf5'
            file_name='dset_{}.hdf5'.format(data_take)
            file_path=experiment_path+file_name
            f=h5py.File(file_path, 'w')
            dset=f.create_dataset('sar_dataset', (self.npos, self.nfre), dtype = np.complex64)
	    
            data=self.vna.send_sweep()
            dset[0,:]=data

            for j in range(1, self.npos):
                if self.stop_flag:
		    break

                self.rail.move(self.dx, 'R')
                data=self.vna.send_sweep()
                dset[j,:]=data

	    if self.stop_flag:
		break
	    
            take_time=datetime.utcnow().replace(tzinf= FROM_ZONE)
            take_time=take_time.astimezone(TO_ZONE)
            dset.attrs['xi']=1.0 * self.xi / METERS_TO_STEPS_FACTOR
            dset.attrs['xf']=1.0 * self.xf / METERS_TO_STEPS_FACTOR
            dset.attrs['dx']=1.0 * self.dx / METERS_TO_STEPS_FACTOR
            dset.attrs['npos']=self.npos
            dset.attrs['fi']=self.fi * 1E9
            dset.attrs['ff']=self.ff * 1E9
            dset.attrs['nfre']=self.nfre
            dset.attrs['ptx']=self.ptx
            dset.attrs['ifbw']=self.ifbw
            dset.attrs['beam_angle']=self.beam_angle
            dset.attrs['datetime']=take_time.strftime("%d-%m-%y %H:%M:%S")
	    f.close()
	    #celery task to send the data to the main server
            send_data.delay(file_path, self.collection_name)
            data_take=data_take+1
	    experiment_collection.update_one({"_id" : "current_experiment"},
					     {"$inc": { "experiment.last_data_take": 1}})
	    #move rail to zero position
	    self.rail.zero()
	self.cleanup()


    def stop(self):
        self.stop_flag=True

    def cleanup(self):
	#on cleanup, will close connections to the rail and vna, will remove the last empty file
	#and will delete the document that holds the 'current_experiment' info
	try:
	    self.rail.close()
	    self.vna.close()
	    f.close()
	    os.remove(file_path)
	    experiment_collection.delete_one({'_id':'current_experiment'})
	except:
	    pass

    def reset_arduino(self):
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(8, GPIO.OUT)
	GPIO.output(8, GPIO.LOW)
	time.sleep(1)
	GPIO.output(8, GPIO.HIGH)
	time.sleep(3)
	return
