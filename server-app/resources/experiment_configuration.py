from flask_restful import Resource
from flask import jsonify, request
from bson import ObjectId

import json, ast
import sys

sys.path.append('/home/Documents/sar-app/')

from common.db import *
from static import sar_experiment

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

class ExperimentConfiguration(Resource):
    def get(self, instruction, config_id = None):
		if instruction == 'list':
			if configuration_collection.find().count() == 0:
				return jsonify("No configurations available")
			else:
				list_of_configurations = list(configuration_collection.find())
				list_of_configurations = JSONEncoder().encode(list_of_configurations)
				return list_of_configurations

		if instruction == 'use' and config_id is not None:
			query = configuration_collection.find_one({"_id" : ObjectId(config_id)})
			post = {'name' : 'current_configuration',
					'config_id' : ObjectId(config_id)}
			configuration_collection.insert_one(post)
			return jsonify(config = config_id)

		if instruction == 'current_configuration':
			query = configuration_collection.find_one({"name" : "current_configuration"})
			config_id = query['config_id']
			current_configuration = configuration_collection.find_one({"_id" : config_id})
			return str(current_configuration)

    def put(self, instruction):
		if instruction == 'new':
			data = request.get_json(force = True)
			new_config = ast.literal_eval(json.dumps(data))

			configuration_collection.insert_one(new_config)
			return jsonify(newconfiguration = data)
