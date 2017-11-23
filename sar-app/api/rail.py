import time
import socket
import logging
import sys

sys.path.append('/home/Documents/sar-app')

from static import constants

RAIL_INSTRUCTIONS = {'move' : '0\n', 'calibrate' : '1\n', 'zero' : '2\n',
                     'stop' : '3\n', 'disconnect' : '4\n'}
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
            #print 'steps out of range (0 - 439 393)'
            return

        if direction is None:
            direction = 'R'
        else:
            direction = direction

        #self.steps = steps
        #self.direction = direction
        self.send(data = RAIL_INSTRUCTIONS['move'] + str(steps) + str(direction) + '\n')
        #print "instruction sent: move %.8f to the right" %(steps*1.0 / METERS_TO_STEPS_FACTOR)
        #print self.recv()

    def stop(self):
        self.send(data = RAIL_INSTRUCTIONS['stop'] + '\n')
        #print "instruction sent: stop"

    '''
    def send_calibrate(self):
        self.send(data = RAIL_INSTRUCTIONS['calibrate'] + '\n')
        print "instruction sent: start calibration"
        #print self.recv()
    '''

    def zero(self):
        self.send(data = RAIL_INSTRUCTIONS['zero'] + '\n')
        #print "instruction sent: move to zero position"
        print self.recv()
        #if int(self.recv()) < 0:
        #    print "error while returning to zero position, please check switch"

    def end_connection(self):
        self.send(data = RAIL_INSTRUCTIONS['disconnect'] + '\n')
        print self.recv()

    def close(self):
        self.socket.close()
        #print "connection to the rail closed"
