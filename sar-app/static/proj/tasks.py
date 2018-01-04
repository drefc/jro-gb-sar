from __future__ import absolute_import
from .celery import app

from celery import Celery

import requests
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from static.constants import UPLOAD_DATA_URL as url
from common.db import *

#bind the task to the calling app
@app.task(bind=True)
def send_data(self, data_path, collection_name):
    headers={'collection_name':collection_name}
    files={'file': open(data_path, 'rb')}
    r=requests

    try:
        r.post(url, files=files, headers=headers)
    except r.ConnectionError as exc:
	#not include max_retries will hold the MaxRetriesExceedError value
	#retry every 10 minutes, for a maximum of 1000 retries
        raise self.retry(countdown=10*60, exc=exc, max_retries=1000)
