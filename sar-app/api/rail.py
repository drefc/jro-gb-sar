import time
import socket
import logging

import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from static import constants

RAIL_INSTRUCTIONS = {'move' : '0\n',
                     'calibrate' : '1\n',
                     'zero' : '2\n',
                     'stop' : '3\n',
                     'disconnect' : '4\n'}

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
            self.close()

    def send(self, data):
        try:
            self.socket.send(data)
        except socket.error:
            #print "could not send instruction to the rail"
            pass

        #if self.error < 0:
        #    print "could not send instruction to the rail"
        #else:
        #    print "instruction sent: %s" %data

    def receive(self):
        data = ''
        aux = ''
        flag = False

        while True:
            aux = self.socket.recv(1)
            if aux == '\n':
                break
            data = data + aux
            #else:
            #    break

            #if flag:
            #    break
        return data

    def move(self, steps, direction = None):
        if 0 < steps < (1450 * 20000 / 66.0):
            steps = steps
        else:
            return

        if direction is None:
            direction = 'R'
        else:
            direction = direction

        self.send(data = RAIL_INSTRUCTIONS['move'] + str(steps) + str(direction) + '\n')
        ack=self.rcv()
        time.sleep(2)
        return

    def stop(self):
        self.send(data = RAIL_INSTRUCTIONS['stop'] + '\n')

    def zero(self):
        self.send(data = RAIL_INSTRUCTIONS['zero'] + '\n')
        ack=self.recv()
        return ack

    def end_connection(self):
        self.send(data = RAIL_INSTRUCTIONS['disconnect'] + '\n')
        ack=self.recv()

    def close(self):
        self.socket.close()
        #print "connection to the rail closed"
