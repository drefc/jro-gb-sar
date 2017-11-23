#!/usr/bin/env python

import paramiko
import sys

from os import listdir
from os.path import isfile, join

sys.path.append('/home/Documents/sar-app/')

from common.db import *
from static.constants import *

class data_sender():
    def __init__(self, folder_name):
        self.folder_name = folder_name

        self.__hostname = '25.56.104.172'
        self.__username = 'andre'
        self.__password = 'soporte'

        self.__destination_folder = DESTINATION_FOLDER
        self.__ssh=paramiko.SSHClient()
        self.__ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def __retrieve_file_list(self):
        if self.__destination_folder not in self.__sftp.listdir():
            self.__sftp.mkdir(self.__destination_folder)

        self.__sftp.chdir(path = self.__destination_folder)

        if self.folder_name not in self.__sftp.listdir():
            self.__sftp.mkdir(self.folder_name)

        self.__sftp.chdir(path = self.folder_name)
        return self.__sftp.listdir()

    def connect(self):
        try:
            self.__ssh.connect(hostname = self.__hostname,
                               username = self.__username,
                               password = self.__password,
                               timeout = 10)
            self.__sftp = self.__ssh.open_sftp()
            return True
        except:
            return False

    def sync_folders(self):
        local_files_list = f for f in listdir(self.source_folder) if isfile(join(self.source_folder, f))

        if set(local_files_list)!=set(self.__retrieve_file_list):
            for element in local_files_list:
                if element not in self.__retrieve_file_list:
                    self.__sftp.put(localpath = element, remotepath = element)

if __name__ == '__main__':
    query = experiment_collection.find_one({"name" : "current_experiment"})

    if query is not None:
        status = current_experiment['status']

        if status is not None:
            while True:
                if status == 'not running':
                    pass

                if status == 'running':
                    folder_name = col['folder_name']
                    data_sender = data_sender(folder_name = folder_name)

                    if data_sender.connect():
                        data_sender.sync_folders()
                    else:
                        #log something
                        pass
    else:
        pass
