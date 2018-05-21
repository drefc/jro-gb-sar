from flask_restful import Resource
from flask import jsonify, request
from datetime import datetime

import os, sys, time

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from common import db
from static.sar_experiment import *
from config import *
from MyGlobals import myglobals

class ExperimentControl(Resource):
	def get(self, instruction):
		if instruction=="start" and myglobals.experiment is None:
			myglobals.experiment=sar_experiment()
			myglobals.experiment.start()
			time.sleep(5)
			return jsonify(status="experiment started")

		if instruction=="stop":
			try:
				if myglobals.experiment.isAlive():
					myglobals.experiment.stop()
					myglobals.experiment=None
					time.sleep(3)
					return jsonify(status="experiment stopped")
			except:
				return jsonify(status="no experiment was running")

		if instruction=="status":
			if myglobals.experiment is None:
				return jsonify(status="no experiment running")
			if myglobals.experiment.isAlive():
				return jsonify(status="experiment running")

