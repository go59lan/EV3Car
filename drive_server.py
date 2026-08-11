

import socket
from ev3dev2.motor import LargeMotor,MediumMotor,  OUTPUT_A, OUTPUT_B, OUTPUT_C, SpeedPercent

steer_motor = MediumMotor(OUTPUT_A)
left_motor = LargeMotor(OUTPUT_B)
right_motor = LargeMotor(OUTPUT_C)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', 5000))

while True:
    data, _ = sock.recvfrom(1024)
    try:
        steering_str, speed_str = data.decode().split(',')
        steering = int(steering_str)
        speed = int(speed_str)

        steer_motor.on_to_position(SpeedPercent(50), steering)

        base_speed = SpeedPercent(speed)
        diff = SpeedPercent(abs(steering) * speed / 100)

        if steering > 0:
            left_motor.on(base_speed)
            right_motor.on(base_speed - diff)
        elif steering < 0:
            left_motor.on(base_speed - diff)
            right_motor.on(base_speed)
        else:
            left_motor.on(base_speed)
            right_motor.on(base_speed)

    except:
        left_motor.off()
        right_motor.off()
        steer_motor.off()

