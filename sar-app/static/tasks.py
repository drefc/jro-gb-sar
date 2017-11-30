from celery import Celery
import requests
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from static.constants import UPLOAD_DATA_URL as url

app = Celery('sar-tasks', broker='pyamqp://guest@localhost//')

@app.task
def send_data(data_path):
    files={'file': open(data_path, 'rb')}
    r=requests.post(url, files=files)
