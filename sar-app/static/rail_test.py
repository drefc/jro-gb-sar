import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from api import rail

rail=rail.railClient()
rail.connect()
rail.move(10000,'L')
rail.end_connection()
rail.close()
