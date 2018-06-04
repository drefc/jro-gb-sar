import os, sys, time

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from api import rail

rail=rail.railClient()
rail.connect()
ack=rail.zero()
print "zero ack: ", ack

'''
for x in range(5):
    start_time=time.time()
    ack=None
    ack=rail.move(60000, 'R')
    print ack
    end_time=time.time()
    print end_time-start_time
    time.sleep(1)
'''

rail.close()
