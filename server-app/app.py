#!/usr/bin/env python

#SIMPLE WEB DATA RECEIVER
#will store data in folder defined by "current_experiment" collection
#in local mongodb

from flask import Flask, current_app
from flask_restful import Resource, Api
from datetime import datetime

from resources.receive_data import ReceiveData
from common.common_functions import *

import logging
import time

app = Flask(__name__)

api = Api(app)
api.add_resource(ReceiveData, '/data_upload')

if __name__ == '__main__':
	host = run_vpn(check_vpn)

	if host:
		app.run(host = host, port = 4500, debug = app.config['DEBUG'])
