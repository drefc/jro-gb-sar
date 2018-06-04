#!/usr/bin/env python
import logging, time, os

from flask import Flask, current_app
from flask_restful import Resource, Api
from datetime import datetime

from common import db
from common.common_functions import *
#from MyGlobals import myglobals

from resources.check_instrument import CheckInstrument
from resources.experiment_configuration import ExperimentConfiguration
from resources.experiment_control import ExperimentControl
from resources.foo import Foo

app=Flask(__name__)
app.config.from_object('config')
app_route='/home/Documents/jro-gb-sar/sar-app'

start_time=datetime.utcnow().replace(tzinfo=app.config['FROM_ZONE'])
start_time=start_time.astimezone(app.config['TO_ZONE'])

log_file_name = '{}.log'.format(str(start_time).replace(' ','_'))
logging.basicConfig(level=logging.DEBUG,
		    format='%(asctime)s %(levelname)s %(message)s',
		    datefmt='%a, %d %b %Y %H:%M:%S',
		    filename=os.path.join(app_route,'log', log_file_name),
		    filemode='w')

logging.info('App started at {}'.format(str(start_time)))

api=Api(app)
api.add_resource(CheckInstrument, '/instrument/check/<string:instrument_name>')
api.add_resource(ExperimentConfiguration, '/configuration/<string:instruction>',
				 '/configuration/<string:instruction>/<config_id>')
api.add_resource(ExperimentControl, '/experiment/<string:instruction>')
api.add_resource(Foo, '/foo')

if __name__ == '__main__':
	while True:
		if check_instruments>0:
			break
		else:
			print "Something is wrong with the rail or vna, retrying in 5 seconds. \
				   If the problem persists, please check the log \
				   file located in the folder /var/log/app."
			time.sleep(5)

	check_existing_experiment()
	host=run_vpn(check_vpn)

	if host:
		app.run(host = host, port = 4500, debug = app.config['DEBUG'])
