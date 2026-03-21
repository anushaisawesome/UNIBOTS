# Try to import RPi.GPIO (only available on Raspberry Pi)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("⚠️  RPi.GPIO not available (expected on Mac, install on Pi with: pip install RPi.GPIO)")
    GPIO_AVAILABLE = False
    # Mock GPIO for development
    class MockGPIO:
        BCM = 'BCM'
        OUT = 'OUT'
        def setmode(self, mode): pass
        def setwarnings(self, state): pass
        def setup(self, pin, mode): pass
        def output(self, pin, state): pass
        class PWM:
            def __init__(self, pin, freq): pass
            def start(self, duty): pass
            def ChangeDutyCycle(self, duty): pass
            def stop(self): pass
    GPIO = MockGPIO()

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# =========================
# MOTOR CLASS
# =========================

class Motor:
    def __init__(self, in1, in2, enable):
        self.in1 = in1
        self.in2 = in2
        self.enable = enable

        GPIO.setup(in1, GPIO.OUT)
        GPIO.setup(in2, GPIO.OUT)
        GPIO.setup(enable, GPIO.OUT)

        self.pwm = GPIO.PWM(enable, 1000)
        self.pwm.start(0)

    def forward(self, speed=100):
        GPIO.output(self.in1, True)
        GPIO.output(self.in2, False)
        self.pwm.ChangeDutyCycle(speed)

    def reverse(self, speed=100):
        GPIO.output(self.in1, False)
        GPIO.output(self.in2, True)
        self.pwm.ChangeDutyCycle(speed)

    def stop(self):
        GPIO.output(self.in1, False)
        GPIO.output(self.in2, False)
        self.pwm.ChangeDutyCycle(0)

# =========================
# PIN CONFIGURATION
# =========================

# Left drive motor
LEFT_IN1 = 5
LEFT_IN2 = 6
LEFT_EN  = 13

# Right drive motor
# FIX: IN1 and IN2 are swapped compared to the left motor because the right
# motor is mounted facing the opposite direction on the chassis. Swapping
# these pins means motor.forward() now drives the right wheel forward (away
# from the robot) instead of backward, so both wheels agree on direction.
RIGHT_IN1 = 26   # ← was 19
RIGHT_IN2 = 19   # ← was 26
RIGHT_EN  = 12

# Spinner motor
# FIX: IN1 and IN2 swapped for the same reason as the right drive motor —
# the spinner is mounted in the opposite orientation so forward() was spinning
# it backwards (away from the collection zone). Swapping these makes
# start_spinner() pull balls inward as intended.
SPIN_IN1 = 21   # ← was 20
SPIN_IN2 = 20   # ← was 21
SPIN_EN  = 16

# Lift motor
LIFT_IN1 = 23
LIFT_IN2 = 24
LIFT_EN  = 18

# =========================
# CREATE MOTOR OBJECTS
# =========================

left_motor  = Motor(LEFT_IN1,  LEFT_IN2,  LEFT_EN)
right_motor = Motor(RIGHT_IN1, RIGHT_IN2, RIGHT_EN)
spinner     = Motor(SPIN_IN1,  SPIN_IN2,  SPIN_EN)
lift        = Motor(LIFT_IN1,  LIFT_IN2,  LIFT_EN)

# =========================
# DRIVE FUNCTIONS
# =========================

def move_forward(speed=80):
    left_motor.forward(speed)
    right_motor.forward(speed)

def move_backward(speed=80):
    left_motor.reverse(speed)
    right_motor.reverse(speed)

def turn_left(speed=80):
    # Left wheel backward, right wheel forward → spins left on the spot
    left_motor.reverse(speed)
    right_motor.forward(speed)

def turn_right(speed=80):
    # Left wheel forward, right wheel backward → spins right on the spot
    left_motor.forward(speed)
    right_motor.reverse(speed)

def turn(left_speed, right_speed):
    """
    Differential steering — pass individual speeds for each wheel.
    Positive = forward, negative = reverse for that wheel.
    Used by steer_toward() in yolo-final.py to curve while moving.
    """
    if left_speed >= 0:
        left_motor.forward(left_speed)
    else:
        left_motor.reverse(-left_speed)

    if right_speed >= 0:
        right_motor.forward(right_speed)
    else:
        right_motor.reverse(-right_speed)

def stop_drive():
    left_motor.stop()
    right_motor.stop()

# =========================
# SPINNER FUNCTIONS
# =========================

def start_spinner(speed=100):
    spinner.forward(speed)

def reverse_spinner(speed=100):
    spinner.reverse(speed)

def stop_spinner():
    spinner.stop()

# =========================
# LIFT FUNCTIONS
# =========================

def lift_up(speed=80):
    lift.forward(speed)

def lift_down(speed=80):
    lift.reverse(speed)

def stop_lift():
    lift.stop()

# =========================
# CLEANUP FUNCTION
# =========================

def shutdown():
    stop_drive()
    stop_spinner()
    stop_lift()
    if GPIO_AVAILABLE:
        GPIO.cleanup()
