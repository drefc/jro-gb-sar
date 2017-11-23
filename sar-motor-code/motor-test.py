from __future__ import division
import os
import numpy as np
import h5py
import rail_ethernet_api
import time

xi = 1000

if __name__ == "__main__":
    rail = rail_ethernet_api.railClient()
    rail.connect()

    for x in range(0,2):
        rail.send_move(xi, 'L')
        time.sleep(0.5)

    rail.end_connection()
    rail.close()
