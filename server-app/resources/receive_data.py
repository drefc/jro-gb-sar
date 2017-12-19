from flask import Flask, jsonify, request
from flask_restful import Resource, reqparse
import werkzeug, json

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from common.common_functions import *
from config import *

class ReceiveData(Resource):
    def post(self):
        parser=reqparse.RequestParser()
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args=parser.parse_args()
        f=args['file']
        collection_name=request.headers['collection_name']

        print collection_name, f
