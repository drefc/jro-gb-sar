from dateutil import tz
import os

FROM_ZONE=tz.gettz('UTC')
TO_ZONE=tz.gettz('America/Lima')
DEBUG=True
DATA_PATH='/home/Documents/data/'
APP_PATH=os.getcwd()
