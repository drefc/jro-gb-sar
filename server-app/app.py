#!/usr/bin/env python

#SIMPLE WEB DATA RECEIVER
#stores data files in folder named after "current_experiment" collection
#appends the stored data in local mongodb for easier retrieving

from flask import Flask
from flask_restful import Resource, Api
from datetime import datetime

from common.common_functions import *
from resources.receive_data import ReceiveData
from resources.foo import Foo

import logging
import time

app = Flask(__name__)

api = Api(app)
api.add_resource(ReceiveData, '/data_upload')
api.add_resource(Foo, '/foo')

if __name__ == '__main__':
	host = run_vpn(check_vpn)

	if host:
		app.run(host = host, port = 4500, debug=False)
