from pymongo import MongoClient

client = MongoClient()
db = client['sar-db']
configuration_collection = db['sar-configuration']
experiment_collection = db['sar-current-experiment']
