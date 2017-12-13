from flask_restful import Resource
from flask import jsonify, request
from datetime import datetime

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from common import db
from static.sar_experiment import *
from config import *
from MyGlobals import myglobals

class ExperimentControl(Resource):
	def get(self, instruction):
		if myglobals.experiment is None:
			msg=str(myglobals.experiment)

		if instruction=='start':
			myglobals.experiment=sar_experiment()
			myglobals.experiment.start()
			msg="experiment started"

		if instruction=='stop':
			if myglobals.experiment.isAlive():
				myglobals.experiment.stop()
				msg="experiment stopped"

		if instruction=='status':
			if myglobals.experiment.isAlive():
				msg="experiment running"
			else:
				msg="no experiment running"
		return jsonify(status=msg)
