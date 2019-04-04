import rail, sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
#   from common.common_functions import reset_arduino

if __name__ == "__main__":
    #reset_arduino()
    rail=rail.railClient()
    rail.connect()
    rail.send_servo_push()
    rail.end_connection()
    rail.close()
