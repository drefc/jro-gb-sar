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

PORT=12345
BUFFER_LENGTH=10000

class railClient():
    def __init__(self, host=None, port=None):
        if host is None:
            host=HOST_LIST['rail']
        if port is None:
            port=PORT
        self.host=host
        self.port=port

    def connect(self):
        self.socket=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.socket.connect((self.host, self.port))
            print "RAIL: rail connected"
            return 1

        except socket.error:
            self.socket.close()
            print "RAIL: rail not connected."
            return -1

    def send(self, data):
        try:
            self.socket.send(data)
        except socket.error:
            print "RAIL: instruction not sent"

    def receive(self, timeout=None):
        data=''
        buff=''

        if timeout:
            start=time.time()
            #self.socket.setblocking(1)

        while True:
            try:
                buff=self.socket.recv(1)
                data=data+buff
                if data=='OK\n':
                    break
            except:
                if timeout:
                    end=time.time()
                    if (end-start)>=timeout:
                        data=-1
                        #self.socket.setblocking(0)
                        break
                else:
                    pass
        print "RAIL: ack received."
        return data

    def move(self, steps, direction=None):
        if direction is None:
            direction='R'
        else:
            direction=direction

        self.send(data=RAIL_INSTRUCTIONS['move']+str(steps)+str(direction)+'\n')
        ack=self.receive(timeout=15)
        #ack=self.receive()
        #time.sleep(2)
        return ack

    def stop(self):
        self.send(data=RAIL_INSTRUCTIONS['stop'] + '\n')

    def zero(self):
        self.send(data=RAIL_INSTRUCTIONS['zero'] + '\n')
        ack=self.receive(timeout=90)
        time.sleep(2)
        return ack

    def disconnect(self):
        self.send(data=RAIL_INSTRUCTIONS['disconnect'] + '\n')

    def close(self):
        self.disconnect()
        self.socket.close()
        print "RAIL: disconnected."
