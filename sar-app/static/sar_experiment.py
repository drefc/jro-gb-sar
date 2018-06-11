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

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.getcwd())

from api import vna, rail
from common.db import *
from config import *
from static import *

class sar_experiment(threading.Thread):
    def __init__(self):
    	#thread constructor
    	self.reset_arduino()
    	#creates rail and vna objects to be available throughout the instance
        self.rail=rail.railClient()
    	self.vna=vna.vnaClient()
        atexit.register(self.cleanup)

        try:
            self.mode=current_configuration['mode']
        except:
            self.mode=None
            pass
        #cleanup before program termination
        threading.Thread.__init__(self)

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

        if current_configuration['beam_angle']:
            self.beam_angle=current_configuration['beam_angle']
        else:
            self.beam_angle=180.0

    	#calculate the 'dx' and 'npos' parameters
    	#recalculate 'xf'
        self.xi=int(self.xi*METERS_TO_STEPS_FACTOR)
        self.xf=int(self.xf*METERS_TO_STEPS_FACTOR)
        self.dx=int((c0/(4.0*self.ff*1E9*np.cos(0.5*(180.0-self.beam_angle)*np.pi/180.0)))*METERS_TO_STEPS_FACTOR)
        self.npos=int(((self.xf-self.xi)/self.dx)+1.0)
        self.xf=self.xi+self.dx*(self.npos-1)
        self.stop_flag = False

    def run(self):
        #connect to the vna
        if self.vna.connect()<0:
            myglobals.status="ERROR: vna connection error."
            self.cleanup()
            return

        self.vna.send_ifbw(self.ifbw)
        self.vna.send_number_points(self.nfre)
        self.vna.send_freq_start(freq_start = self.fi)
        self.vna.send_freq_stop(freq_stop = self.ff)
        self.vna.send_power(self.ptx)
        self.vna.send_select_instrument()
        self.vna.send_cfg()

        if self.rail.connect()<0:
            myglobals.status="ERROR: rail connection error."
            self.cleanup()
            return

    	if self.rail.zero()<0:
            myglobals.status="ERROR: check rail."
            self.cleanup()
            return
        #find if there was an existing experiment running
        try:
            query=experiment_collection.find_one({"_id":"current_experiment"})
        except:
            myglobals.status="ERROR: db error."
            self.cleanup()
            return
        #if there was, search for the 'last_data_take' and the 'folder_name' variables
        #last_data_take: stores the last data take that was FULLY COMPLETED
        #folder_name: folder where the data from the current experiment is being stored
        if query:
            data_take=query['experiment']['last_data_take']
            folder_name=str(query['experiment']['folder_name'])
            experiment_path=os.path.join(DATA_PATH, folder_name)
        else:
        #if there was not, create a new folder, and append the current datetime to avoid conflicts with repeated 'collection_name' parameters
        #update (or create if not exists) the 'current_experiment' collection
            start_time=datetime.utcnow().replace(tzinfo=FROM_ZONE)
            start_time=start_time.astimezone(TO_ZONE)
            folder_name="{}_{}".format(self.collection_name, start_time.strftime("%d-%m-%y_%H:%M:%S"))
            experiment_path=os.path.join(DATA_PATH, folder_name)

            os.mkdir(experiment_path)
            #PENDING: should log the folder creation
            data_take=1

            experiment={"folder_name":folder_name,
                        "last_data_take":data_take}
            experiment_collection.find_one_and_update({"_id":"current_experiment"},
                                                      {"$set":{"experiment": experiment}},
                                                      upsert=True)
        #the experiment loop begins
        #this will loop infinitely until an 'stop' is requested, or there is a power failure and the system needs to go down

        if self.mode=="continuous-mode":
            self.rail.close()
            myglobals.status="experiment running."

            while True:
                self.rail=rail.rail_continuous()
                self.rail.connect()
                self.rail.zero()
                #the datasets will be named 'dset_{data_take}.hdf5'
                t=[]
                d=[]
                self.rail.start()

                while True:
                    d.append(self.vna.send_sweep())
                    t.append(time.time())
            	    if self.rail.get_status():
            		    break
                    if self.stop_flag:
                        break

                if self.stop_flag:
                    break

                data=np.array(d)
                timestamp=np.array(t)

                file_name='dset_{}.hdf5'.format(data_take)
                self.file_path=os.path.join(experiment_path, file_name)
                self.f=h5py.File(self.file_path, 'w')
                dset=self.f.create_dataset('sar_dataset', data.shape, dtype=np.complex64)

        	    take_time=datetime.utcnow().replace(tzinfo=FROM_ZONE).astimezone(TO_ZONE)
                dset.attrs['take_index']=data_take
                dset.attrs['xi']=str(1.0 * self.xi / METERS_TO_STEPS_FACTOR)
                dset.attrs['xf']=str(1.0 * self.get_aperture_length / METERS_TO_STEPS_FACTOR)
                dset.attrs['npos']=data.shape[1]
                dset.attrs['fi']=self.fi * 1E9
                dset.attrs['ff']=self.ff * 1E9
                dset.attrs['nfre']=self.nfre
                dset.attrs['ptx']=self.ptx
                dset.attrs['ifbw']=self.ifbw
                dset.attrs['beam_angle']=self.beam_angle
                dset.attrs['datetime']=take_time.strftime("%d-%m-%y %H:%M:%S")
        	    self.f.close()
        	    #celery task to send the data to the main server
                send_data.delay(file_name, self.file_path, folder_name)
                data_take=data_take+1
        	    experiment_collection.update_one({"_id" : "current_experiment"},
        					     {"$inc": { "experiment.last_data_take": 1}})
        	    #move rail to zero position
                self.rail.close()

        if self.mode==None:
            while True:
                #move to the starting position (only if it is different from 0)
                myglobals.status="experiment running."
                if self.xi!=0:
                    if self.rail.move(self.xi, 'R')<0:
                        myglobals.status="ERROR: check rail."
                        break
                file_name='dset_{}.hdf5'.format(data_take)
                self.file_path=os.path.join(experiment_path, file_name)
                self.f=h5py.File(self.file_path, 'w')
                dset=self.f.create_dataset('sar_dataset', (self.npos, self.nfre), dtype=np.complex64)

                dset[0,:]=self.vna.send_sweep()

                for j in range(1, self.npos):
                    if self.stop_flag:
    		            break

                    self.rail.move(self.dx, 'R')
                    dset[j,:]=self.vna.send_sweep()

        	    if self.stop_flag:
        		    break

        	    take_time=datetime.utcnow().replace(tzinfo=FROM_ZONE).astimezone(TO_ZONE)
                dset.attrs['take_index']=data_take
                dset.attrs['xi']=str(1.0 * self.xi / METERS_TO_STEPS_FACTOR)
                dset.attrs['xf']=str(1.0 * self.xf / METERS_TO_STEPS_FACTOR)
                dset.attrs['dx']=str(1.0 * self.dx / METERS_TO_STEPS_FACTOR)
                dset.attrs['npos']=self.npos
                dset.attrs['fi']=self.fi * 1E9
                dset.attrs['ff']=self.ff * 1E9
                dset.attrs['nfre']=self.nfre
                dset.attrs['ptx']=self.ptx
                dset.attrs['ifbw']=self.ifbw
                dset.attrs['beam_angle']=self.beam_angle
                dset.attrs['datetime']=take_time.strftime("%d-%m-%y %H:%M:%S")
        	    self.f.close()
        	    #celery task to send the data to the main server
                send_data.delay(file_name, self.file_path, folder_name)
                data_take=data_take+1
        	    experiment_collection.update_one({"_id" : "current_experiment"},
        					     {"$inc": { "experiment.last_data_take": 1}})
        	    #move rail to zero position
                if self.rail.zero()<0:
                    myglobals.status="ERROR: check rail."
                    break

    	self.cleanup()
        myglobals.status="experiment not running."

    def stop(self):
        self.stop_flag=True

    def cleanup(self):
	#on cleanup, will close connections to the rail and vna, will remove the last empty file
	#and will delete the document that holds the 'current_experiment' info
    	try:
    	    self.rail.close()
    	except:
    	    print "CLEANUP: rail was not connected."

    	try:
        	self.vna.close()
    	except:
    	    print "CLEANUP: vna was not connected."

    	try:
    	    self.f.close()
    	except:
    	    print "CLEANUP: file was not openned."

        '''
    	try:
            experiment_collection.find_one_and_delete({"_id":"current_experiment"})
    	except:
    	    print "CLEANUP: current experiment collection not deleted."
        '''

    def reset_arduino(self):
    	GPIO.setwarnings(False)
    	GPIO.setmode(GPIO.BOARD)
    	GPIO.setup(18, GPIO.OUT)
    	GPIO.output(18, GPIO.LOW)
    	time.sleep(1)
    	GPIO.output(18, GPIO.HIGH)
    	time.sleep(3)
