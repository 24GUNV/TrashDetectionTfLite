# File to turn on/off the lights

# import libraries
import wiringpi as wiringpi

# Initialize GPIO mode
wiringpi.wiringPiSetupGpio()

# Seting up ports for input/output
wiringpi.pinMode(5,1)

try:
    while True:
         wiringpi.digitalWrite(5, 1)
except KeyboardInterrupt: # When they click Ctrl + C
    wiringpi.digitalWrite(5, 0)
    wiringpi.pinMode(5, 0)
    print("Ending Program")