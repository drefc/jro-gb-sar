import os, sys, time, threading

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from api import rail

while True:
    r=rail.rail_continuous()
    r.connect()
    r.zero()
    r.close()

'''
for x in range(3):
    r=rail.rail_continuous()
    r.connect()
    r.start()

    while True:
        status=r.get_status()
        print status
        if status:
            break
        time.sleep(2)

    r.close()
    r=None

ack=rail.zero()
print "zero ack: ", ack

for x in range(5):
    start_time=time.time()
    ack=None
    ack=rail.move(60000, 'R')
    print ack
    end_time=time.time()
    print end_time-start_time
    time.sleep(1)

rail.close()
'''
