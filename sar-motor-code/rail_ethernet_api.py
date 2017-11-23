import time
import socket
import sys
import string
import os
import numpy as np
from constants import *

ARDUINO_INSTRUCTIONS = {'move' : '0',
                        'calibrate' : '1',
                        'zero_position' : '2',
                        'stop' : '3',
                        'end_connection' : '4'}

HOST = '10.10.40.245'
#HOST = '10.10.50.236'
PORT = 12345
BUFFER_LENGTH = 10000

class railClient():
    def __init__(self, host = None, port = None):
        if host is None:
            host = HOST
        if port is None:
            port = PORT
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5)

    def connect(self):
        try:
            self.socket.connect((self.host, self.port))
        except socket.error:
            self.socket.close()
            self.socket = None

        if self.socket is None:
            print "could not connect to the rail"
        else:
            print "connected to the rail"

    def send(self, data):
        self.error = 0

        try:
            self.socket.send(data)
        except socket.error:
            self.error = -1

        if self.error < 0:
            print "could not send data: %s" %data
        #else:
        #    print "instruction sent: %s" %data

    def recv(self):
        self.data = ''
        aux = ''
        flag = False
        x = ''

        while True:
            aux = self.socket.recv(1)
            if aux:
                for x in aux:
                    self.data += x
                    if x == '\n':
                        flag = True
            else:
                break

            if flag:
                break
        return self.data

    def send_move(self, steps, direction = None):
        if 0 < steps < (1450 * 20000 / 66.0):
            steps = steps
        else:
            print 'steps out of range (0 - 439 393)'
            return

        if direction is None:
            direction = 'R'
        else:
            direction = direction

        self.steps = steps
        self.direction = direction
        self.send(data = ARDUINO_INSTRUCTIONS['move'] + str(self.steps) + self.direction + '\n')
        print "instruction sent: move %.8f to the right" %(steps*1.0 / METERS_TO_STEPS_FACTOR)
        #print self.recv()

    def send_stop(self):
        self.send(data = ARDUINO_INSTRUCTIONS['stop'] + '\n')
        print self.recv()
        print "instruction sent: stop"

    def send_calibrate(self):
        self.send(data = ARDUINO_INSTRUCTIONS['calibrate'] + '\n')
        print "instruction sent: start calibration"
        #print self.recv()

    def send_zero_position(self):
        self.send(data = ARDUINO_INSTRUCTIONS['zero_position'] + '\n')
        print "instruction sent: move to zero position"
        print self.recv()
        #if int(self.recv()) < 0:
        #    print "error while returning to zero position, please check switch"

    def end_connection(self):
        self.send(data = ARDUINO_INSTRUCTIONS['end_connection'] + '\n')

    def close(self):
        self.socket.close()
        print "connection to the rail closed"
