import time
import socket
import logging

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from static.constants import *

RAIL_INSTRUCTIONS = {'move' : '0',
                     'zero' : '2',
                     'stop' : '3',
                     'disconnect' : '4'}

PORT = 12345
BUFFER_LENGTH = 10000

log_path = '../log/'

class railClient():
    def __init__(self, host = None, port = None):
        if host is None:
            host = HOST_LIST['rail']
        if port is None:
            port = PORT
        self.host = host
        self.port = port

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.socket.connect((self.host, self.port))
        except socket.error:
            self.socket.close()

    def send(self, data):
        try:
            self.socket.send(data)
        except socket.error:
            print "could not send instruction to the rail"

    def receive(self):
        data=''
        buff=''

        while True:
            buff=self.socket.recv(1)
            if buff=='\n':
                break
            data=data+buff
        return data

    def move(self, steps, direction=None):
        if direction is None:
            direction='R'
        else:
            direction=direction

        self.send(data=RAIL_INSTRUCTIONS['move']+str(steps)+str(direction)+'\n')
        ack=self.receive()
        time.sleep(2)

    def stop(self):
        self.send(data = RAIL_INSTRUCTIONS['stop'] + '\n')

    def zero(self):
        self.send(data = RAIL_INSTRUCTIONS['zero'] + '\n')
        ack=self.receive()

    def end_connection(self):
        self.send(data = RAIL_INSTRUCTIONS['disconnect'] + '\n')
        ack=self.receive()

    def close(self):
        self.socket.close()
