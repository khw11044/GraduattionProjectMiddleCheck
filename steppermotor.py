from adafruit_motor import stepper

kit = MotorKit()

for i in range(200):
    kit.stepper1.onestep(DIRECTION=STEPPER.backward,style=stepper.DOUBLE)