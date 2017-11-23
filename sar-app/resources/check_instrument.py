from flask_restful import Resource
from flask import jsonify

import sys

sys.path.append('/home/Documents/sar-app/')

from common.common_functions import *
from static.constants import *

class CheckInstrument(Resource):
	def get(self, instrument_name):
		status = 'online' if ping(HOST_LIST[instrument_name]) else 'offline'
		return jsonify(instrument_name = instrument_name, status = status)
