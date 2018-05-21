from flask import Flask, jsonify, request
from flask_restful import Resource, reqparse
import werkzeug, json

import os, sys, h5py

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.getcwd())

from common.common_functions import *
from common.db import *
import config

class ReceiveData(Resource):
    def post(self):
        parser=reqparse.RequestParser()
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args=parser.parse_args()
        data_file=args['file']
        collection_name=request.headers['collection_name']
        upload_path=os.path.join(config.UPLOAD_FOLDER, collection_name)

        if not os.path.isdir(upload_path):
            os.mkdir(upload_path)

        file_location=os.path.join(upload_path, data_file.filename)
        data_file.save(file_location)
        self.insert_file(collection_name, file_location)

    def insert_file(self, collection_name, file_location):
        tmp=h5py.File(file_location, 'r+')
        dset=tmp['sar_dataset']

        #insert metadata to the collection (if does not exist)
        if not db[collection_name].find_one({'_id':'config'}):
            post={'_id':'config',
                  'start_position':dset.attrs['xi'],
                  'stop_position':dset.attrs['xf'],
                  'npos':int(dset.attrs['npos']),
                  'delta':dset.attrs['dx'],
                  'start_freq':float(dset.attrs['fi']),
                  'stop_freq':float(dset.attrs['ff']),
                  'nfre':int(dset.attrs['nfre']),
                  'beam_angle':int(dset.attrs['beam_angle'])}
            db[collection_name].insert_one(post)
        #insert data to the collection
        post={'type':'data',
              'path':file_location,
              'datetime':dset.attrs['datetime'],
              'take_index':int(dset.attrs['take_index'])}
        db[collection_name].insert_one(post)
        tmp.close()
