import os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from api import rail

rail=rail.railClient()
rail.connect()
<<<<<<< HEAD
rail.move(10000,'R')
=======
rail.move(10000)
>>>>>>> 190ab771163a48bd32e6ec7e1ccb1d6d7036ff11
rail.end_connection()
rail.close()
