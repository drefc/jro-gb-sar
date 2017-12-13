from flask_restful import Resource
from flask import jsonify, request
from bson import ObjectId

import json, ast
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from common.db import *
from static import sar_experiment

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

class ExperimentConfiguration(Resource):
    def get(self, instruction, config_id = None):
		if instruction=='current_configuration':
			current_configuration=configuration_collection.find_one({"_id":"current_configuration"})
			return jsonify(current_configuration=str(current_configuration))

    def put(self, instruction):
		if instruction=='set_configuration':
			data=request.get_json(force=True)

			new_config=ast.literal_eval(json.dumps(data))

			configuration=configuration_collection.find_one_and_update({"_id":"current_configuration"},
                                                                       {"$set":{"configuration": new_config}},
                                                                       projection={"configuration":True, "_id":False},
                                                                       upsert=True)
			return jsonify(inserted_configuration=str(configuration))
