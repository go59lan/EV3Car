import socket
from ev3dev2.motor import (
    LargeMotor,
    MediumMotor,
    OUTPUT_A,
    OUTPUT_B,
    OUTPUT_C,
    SpeedPercent,
)

steer_motor = MediumMotor(OUTPUT_A)
left_motor = LargeMotor(OUTPUT_B)
right_motor = LargeMotor(OUTPUT_C)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(2.0)
sock.bind(("", 5000))

print("EV3 drive server listening on UDP port 5000")

try:
    while True:
        data, address = sock.recvfrom(1024)

        try:
            steering_str, speed_str = data.decode().strip().split(",")

            # Protect the motors from invalid controller values.
            # Steering can go up to +-130 (robot_controller2.py's
            # clamp) - the tightest curve on the track needs more
            # than +-100 to actually complete the turn.
            steering = max(-130, min(130, int(steering_str)))
            speed = -max(0, min(100, int(speed_str)))

            # Move the physical steering motor without blocking drive updates.
            steer_motor.on_to_position(
                SpeedPercent(50),
                steering,
                block=False,
            )

            # At steering:
            #   0    -> inner wheel = speed (same as outer)
            #   65   -> inner wheel = speed / 2
            #   130  -> inner wheel = 0
            turn_amount = abs(steering) / 130.0
            inner_speed = round(speed * (1.0 - turn_amount))

            if steering > 0:
                # Right turn: right wheel is the inner wheel.
                left_speed = speed
                right_speed = inner_speed

            elif steering < 0:
                # Left turn: left wheel is the inner wheel.
                left_speed = inner_speed
                right_speed = speed

            else:
                left_speed = speed
                right_speed = speed

            left_motor.on(SpeedPercent(left_speed))
            right_motor.on(SpeedPercent(right_speed))

        except (ValueError, UnicodeDecodeError) as error:
            left_motor.off(brake=True)
            right_motor.off(brake=True)

except KeyboardInterrupt:
    print("Stopping EV3 drive server")

finally:
    left_motor.off(brake=True)
    right_motor.off(brake=True)
    steer_motor.off(brake=True)
    sock.close()
