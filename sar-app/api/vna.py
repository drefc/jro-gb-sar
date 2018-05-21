import time, socket, sys, string, os
import numpy as np

HOST='10.10.40.200'
PORT=9001
IFBW_VALUES=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
             10000, 20000, 50000, 100000]
INSTRUMENT={'SpectrumAnalyzer': 'SPA', 'HighAccuracyPowerMeter': 'HI_PM',
            'InterfaceAnalysis': 'IA', 'ChannelScanner': 'CS',
            'NetworkAnalyzer': 'MWVNA', 'AMFMPM': 'AMFMPM',
            'PowerMonitor': 'Power Monitor', 'VectorVoltmeter': 'VVM'}
BUFFER_LENGTH=100

SCPI_IDN="*IDN?\r\n"
SCPI_RESET="*RST\r\n"
SCPI_NUMBER_OF_POINTS=":SENS:SWE:POIN %d\r\n"
SCPI_IFBW="SENS:SWE:IFBW %d\r\n"
SCPI_FREQ_START=":SENS:FREQ:STAR %.3fGHZ\r\n"
SCPI_FREQ_STOP=":SENS:FREQ:STOP %.3fGHZ\r\n"
SCPI_POWER="SOUR:POW %s\r\n"
SCPI_SELECT_INSTRUMENT="INST:SEL %s\r\n"

SCPI_DATA_FORMAT=":FORM REAL,64\r\n"
SCPI_SWEEP_TYPE=":SENS:SWE:TYPE SING\r\n"
SCPI_INIT_OFF=":INIT:CONT OFF\r\n"
SCPI_INIT_IMM=":INIT:IMM\r\n"

SCPI_TRACE_NUMBER=":SENS:TRAC:TOT %d\r\n"
SCPI_TRACE_SELECT=":SENS:TRAC%d:SEL\r\n"
SCPI_TRACE_DOMAIN=":SENS:TRAC%d:DOM FREQ\r\n"
SCPI_TRACE_PARAMETER=":SENS:TRAC%d:SPAR %s\n"
SCPI_DISPLAY=":DISP:TRAC:FORM %s\r\n" #SINGle, DUAL, TRI,QUAD
SCPI_TRACE_FORMAT=":CALC%d:FORM %s\r\n" #LMAG, SWR, PHA, REAL, IMAG,

SCPI_STATUS_OPERATION=":STAT:OPER?\r\n"
SCPI_TRANSFER_DATA=":CALC:DATA? SDAT\r\n"

class vnaClient():
    def __init__(self, host=None, port=None):
        if host is None:
            host=HOST
        if port is None:
            port=PORT
        self.host=host
        self.port=port

    def connect(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.s.connect((self.host, self.port))
        except socket.error:
                self.s.close()
                self.s = None

        if self.s is None:
            print "vna: could not connect to VNA"
        else:
            print "vna: connected"

    def send(self, data):
        self.error=0

        try:
            self.s.send(data)
        except socket.error:
            self.error=-1
	
	'''
        if self.error<0:
            print "could not send instruction"
        else:
            print 'Data sent: %s\n' % data
	'''

    def recv(self, buffer_len=None):
        if buffer_len is None:
            recv_flag=False
            buffer_len=BUFFER_LENGTH
        else:
            recv_flag=True
	    buffer_len=buffer_len

        data=""

        while True:
            data=data+self.s.recv(1)	    
	    #print "data: {}".format(data)
	    #print "len: {}".format(len(data))
            if recv_flag and len(data)==buffer_len:
                break
            elif not recv_flag and data[len(data)-1]=='\n':
                break

        return data

    def send_idn(self):
        self.send(data = SCPI_IDN)
        #print self.recv(33)

    def send_ifbw(self, ifbw = None):
        if ifbw is None:
            ifbw = IFBW_VALUES[6]
        else:
            if ifbw in IFBW_VALUES:
                ifbw = ifbw
            else:
                print "ifbw not valid"
                return
        self.ifbw = ifbw
        self.send(data = SCPI_IFBW %self.ifbw)

    def send_number_points(self, points = None):
        if points is None:
            points = 1601
        else:
            if 2 <= points <= 4001:
                points = points
            else:
                print "number of points out of range (2-4001)"
                return
        self.points = points
        self.send(data = SCPI_NUMBER_OF_POINTS %self.points)

    def send_freq_start(self, freq_start = None):
        if freq_start is None:
            freq_start = 15.5
        else:
            if 12.4 <= freq_start <= 18.0:
                freq_start = freq_start
            else:
                print "frequency out of range (12.4-18.0 GHz)"
                return
        self.freq_start = freq_start
        #print freq_start
        self.send(data = SCPI_FREQ_START %self.freq_start)

    def send_freq_stop(self, freq_stop = None):
        if freq_stop is None:
            freq_stop = 16.5
        else:
            if 12.4 <= freq_stop <= 18.0:
                freq_stop = freq_stop
            else:
                print "frequency out of range (12.4-18.0 GHz)"
                return
        self.freq_stop = freq_stop
        #print freq_stop
        self.send(data = SCPI_FREQ_STOP %self.freq_stop)

    def send_power(self, power = None):
        if power is None:
            power = 'LOW'
        else:
            if power in ['HIGH', 'LOW']:
                power = power
            else:
                print "invalid power value (HIGH or LOW)"
                return
        self.power = power
        self.send(data = SCPI_POWER %self.power)

    def send_select_instrument(self, instrument = None):
        if instrument is None:
            instrument = INSTRUMENT['NetworkAnalyzer']
        else:
            if instrument in INSTUMENT:
                instrument = INSTRUMENT[instrument]
            else:
                print "instrument not valid"
                return
        self.instrument=instrument
        self.send(data=SCPI_SELECT_INSTRUMENT %self.instrument)

    def send_cfg(self):
        #DISPLAY CONFIGURATION
        #Two traces on screen for the s21 parameter:
        #First for the magnitude (dB) and the other for the phase (rad)
        self.send(data=SCPI_TRACE_NUMBER %2)
        self.send(data=SCPI_DISPLAY %("DUAL"))
        self.send(data=SCPI_TRACE_DOMAIN %1)
        self.send(data=SCPI_TRACE_DOMAIN %2)
        self.send(data=SCPI_TRACE_PARAMETER %(1, "S21"))
        self.send(data=SCPI_TRACE_PARAMETER %(2, "S21"))
        self.send(data=SCPI_TRACE_FORMAT %(1, "LMAG"))
        self.send(data=SCPI_TRACE_FORMAT %(2, "PHAS"))
        self.send(data=SCPI_SWEEP_TYPE)
        self.send(data=SCPI_INIT_OFF)
        time.sleep(5)
        return
        #print "vna succesfuly configured!"

    def send_sweep(self):
        #Configure the data format as REAL,64
        #Set single sweeps
        #Do a single sweep
        self.send(data=SCPI_INIT_IMM)
        #Wait for the insrument to be ready to send the measured data
	time.sleep(1)

        while True:
            self.send(data=SCPI_STATUS_OPERATION)
	    data=self.recv()
	    #print data
	    if data=="256\n":
	        break
	    #time.sleep(0.5)
        
        self.send(data=SCPI_DATA_FORMAT)
        self.send(data=SCPI_TRANSFER_DATA)
	#time.sleep(2)
        while True:
            #x = self.recv(1)
            #print x
	    if self.recv(1)=="#":
            #if x == "#":
                break
        BYTE_COUNT=self.recv(1)
        print BYTE_COUNT
        DATA_BYTE=self.recv(int(BYTE_COUNT))
        print int(DATA_BYTE)
        DATA=self.recv(int(DATA_BYTE))
        print len(DATA)

        data_array=np.fromstring(DATA, dtype=np.complex64)
	time.sleep(2)
        #print data_array.shape
        print "vna: data received!"
        return data_array

    def close(self):
        self.s.close()
        print "vna: connection closed"
