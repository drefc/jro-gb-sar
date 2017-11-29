from celery import Celery
import requests

app = Celery('sar-tasks', broker='pyamqp://guest@localhost//')

@app.task
def send_data(data_path):
    files={'file': open(data_path, 'rb')}
    r=requests.post(url, files=files)
