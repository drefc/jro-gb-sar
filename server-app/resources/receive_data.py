from flask import Flask, jsonify, request
from flask_restful import Resource, reqparse
import werkzeug, json

import os, sys, h5py

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from common.common_functions import *
import config

class ReceiveData(Resource):
    def post(self):
        parser=reqparse.RequestParser()
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args=parser.parse_args()
        f=args['file']
        collection_name=request.headers['collection_name']
        upload_path=config.UPLOAD_FOLDER+collection_name

        if not os.path.isdir(upload_path):
            os.mkdir(upload_path)

        file_route=self.save_file(f,upload_path+'/')
        self.insert_file(collection_name, file_route)

    def save_file(self, myfile, path):
        myfile.save(path+myfile)
        return path+myfile

    def insert_file(self, collection_name, myfile):
        tmp=h5py.File(myfile, 'r+')
        
