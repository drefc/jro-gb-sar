#!/usr/bin/env python

from flask import Flask, current_app
from flask_restful import Resource, Api
from datetime import datetime

from common import db
from common.common_functions import *
#from MyGlobals import myglobals

from resources.check_instrument import CheckInstrument
from resources.experiment_configuration import ExperimentConfiguration
from resources.experiment_control import ExperimentControl

import logging
import time

app = Flask(__name__)
app.config.from_object('config')

'''
start_time = datetime.utcnow().replace(tzinfo = app.config['FROM_ZONE'])
start_time = start_time.astimezone(app.config['TO_ZONE'])

log_file_name = str(start_time).replace(' ','_') + '.log'
logging.basicConfig(level = logging.DEBUG,
		    format = '%(asctime)s %(levelname)s %(message)s',
		    datefmt='%a, %d %b %Y %H:%M:%S',
		    filename = '/home/Documents/sar-app/log/' + log_file_name,
		    filemode = 'w')

logging.info('App started running at ' + str(start_time))
'''

api = Api(app)
api.add_resource(CheckInstrument, '/instrument/check/<string:instrument_name>')
api.add_resource(ExperimentConfiguration, '/configuration/<string:instruction>',
				 '/configuration/<string:instruction>/<config_id>')
api.add_resource(ExperimentControl, '/experiment/<string:instruction>')

if __name__ == '__main__':
	check_existing_experiment()
	host = run_vpn(check_vpn)

	if host:
		app.run(host = host, port = 4500, debug = app.config['DEBUG'])
