from __future__ import division
import rail
import time
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

if __name__ == "__main__":
    rail = rail.railClient()
    rail.connect()
    rail.send_servo_push()
    rail.send_servo_push()
    rail.end_connection()
    rail.close()
