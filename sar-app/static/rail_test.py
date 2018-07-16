from constants import *
from parameters import *
import rail_api

rail = rail_api.railClient()
rail.connect()
#rail.send_zero_position()
#rail.send_move(60000, 'R')

rail.end_connection()
rail.close()
