from flask_restful import Resource
from flask import jsonify, request

import sys

sys.path.append('/home/Documents/sar-app')

from MyGlobals import myglobals

class InstanceTest(Resource):
    def get(self, instruction):
        if instruction=='hola':            
            myglobals.counter=myglobals.counter+1

        if instruction=='chau':
            pass
