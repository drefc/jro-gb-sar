import time, socket, threading, os, sys

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

    def receive(self, timeout=None, continuous=False):
        data=''
        buff=''

        if timeout:
            start=time.time()

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
                        break
                else:
                    pass
        print "RAIL: ack received."
        return True

    def move(self, steps, direction=None, continuous=False):
        if direction is None:
            direction='R'
        else:
            direction=direction

        self.send(data=RAIL_INSTRUCTIONS['move']+str(steps)+str(direction)+'\n')

        if continuous:
            ack=self.receive()
        else:
            ack=self.receive(timeout=15)
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

class rail_continuous(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.rail=railClient()
        self.status=False
        self.aperture_length=int(1.45*METERS_TO_STEPS_FACTOR)

    def connect(self):
        self.rail.connect()

    def close(self):
        self.rail.close()

    def zero(self):
        return self.rail.zero()

    def get_status(self):
        return self.status

    def change_aperture_length(self, aperture_length):
        self.aperture_length=aperture_length

    def get_aperture_length(self):
        return self.aperture_length / (METERS_TO_STEPS_FACTOR*1.0)

    def run(self):
        if self.rail.move(steps=self.aperture_length, continuous=True):
            self.status=True
