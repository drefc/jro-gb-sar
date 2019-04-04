from os import system as system_call
from platform import system as system_name

import netifaces, db, os, sys, time

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from MyGlobals import myglobals
from common.db import *
from static.constants import *

###############################################################################
######################ANOTHER BIG UPDATE!!!####################################
###############################################################################
from api import rail
###############################################################################
######################ANOTHER BIG UPDATE!!!####################################
###############################################################################

def ping(host):
    parameters = "-n 1" if system_name().lower()=="windows" else "-cq 1"
    return system_call("ping " + parameters + " " + host) == 0

def get_dictionary(list, name):
    return filter(lambda dictionary: dictionary['name'] == name, list)[0]

def update_dictionary_value(list, name, key, new_value):
	dictionary = get_dictionary(list, name)
	dictionary[key] = new_value

def run_vpn(cb):
	while True:
		try:
			netifaces.ifaddresses('ham0')
			r = cb(netifaces.ifaddresses('ham0'))
			if r[0]:
				return r[1]
		except Exception:
		    pass

def check_vpn(address):
	host = str(address[netifaces.AF_INET][0]['addr'])

	try:
		ping(host)
		return (True, host)
	except Exception as e:
		pass
	return (False,)

def check_existing_experiment():
    query=configuration_collection.find_one({"_id":"current_configuration"})

    if query:
        query=experiment_collection.find_one({"_id" : "current_experiment"})
        if query:
            status=query['status']
            if status=='running':
                myglobals.experiment=sar_experiment()
                myglobals.experiment.start()
    else:
        myglobals.status="experiment not running."

###############################################################################
######################ANOTHER BIG UPDATE!!!####################################
###############################################################################

def check_instruments():
    if not ping(HOST_LIST['rail']):
        print "rail is not online, rebooting arduino"
        reset_arduino()
        return -1
    else:
        print "rail is online"

    if not ping(HOST_LIST['vna']):
        print "vna is not online, turning on VNA"
        rail=rail.railClient()
        rail.connect()
        print rail.servo_push()
        print "waiting for the vna..."
        rail.close()
        time.sleep(10)
        return -1
    else:
        print "vna is online"

    return 1

###############################################################################
######################ANOTHER BIG UPDATE!!!####################################
###############################################################################
"""
def reset_arduino():
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(12, GPIO.OUT)
    GPIO.output(12, GPIO.LOW)
    time.sleep(1)
    GPIO.cleanup(12)
    #GPIO.output(12, GPIO.HIGH)
    time.sleep(5)
"""
