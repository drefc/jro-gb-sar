from flask import Flask, jsonify
from flask_restful import Resource, reqparse
import werkzeug

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from common.common_functions import *

class ReceiveData(Resource):
    def post(self):
        parse=reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        my_file = args['file']
        #define where to store the files here!
        #my_file.save("your_file_name.jpg")
