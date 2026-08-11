import cv2
import socket

EV3_IP = '10.42.0.3'
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Kp = 0.5
# Ki = 0.01
# Kd = 0.1

# integral = 0
# last_error = 0



class RobotController:
    def __init__(self):
        self.cap = None
        self.latest_frame = None
        self.running = False
        self.Kp = 0.5
        self.Ki = 0.01
        self.Kd = 0.1
        self.integral = 0
        self.last_error = 0

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("No camera frame")
                continue

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            roi = thresh[int(h * 0.85):int(h * 0.95), :]

            m = cv2.moments(roi)
            cx = int(m["m10"] / m["m00"]) if m["m00"] != 0 else w // 2
            error = (w // 2) - cx
            self.integral += error
            derivative = error - self.last_error
            correction = int(self.Kp * error + self.Ki * self.integral + self.Kd * derivative)
            correction = max(min(correction, 100), -100)
            self.last_error = error

            speed = 20
            msg = f"{correction},{speed}".encode()
            sock.sendto(msg, (EV3_IP, 5000))
            print(f"sending to ev3 correction={correction}, error={error}")

            # תצוגה גרפית:
            cv2.line(frame, (w // 2, h), (w // 2, h - 30), (255, 0, 0), 2)     # קו אמצע
            cv2.circle(frame, (cx, int(h * 0.9)), 8, (0, 255, 0), -1)
        # מיקום קו מזוהה
            cv2.putText(frame, f"Error: {error}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Correction: {correction}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


            success, jpeg = cv2.imencode(".jpg", frame)
            if success:
                self.latest_frame = jpeg.tobytes()
            # cv2.imshow("Camera + Path Detection", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        
        self.cap.release()
        cv2.destroyAllWindows()

# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("מצלמה לא נפתחה!")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("אין פריים מהמצלמה")
#         continue

#     h, w = frame.shape[:2]
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
#     roi = thresh[int(h * 0.8), :]

#     m = cv2.moments(roi)
#     cx = int(m["m10"] / m["m00"]) if m["m00"] != 0 else w // 2
#     error = cx - w // 2
#     integral += error
#     derivative = error - last_error
#     correction = int(Kp * error + Ki * integral + Kd * derivative)
#     correction = max(min(correction, 100), -100)
#     last_error = error

#     speed = 40
#     msg = f"{correction},{speed}".encode()
#     sock.sendto(msg, (EV3_IP, 5000))
#     print(f"שולח ל-EV3: תיקון={correction}, מהירות={speed}")

#     # תצוגה גרפית:
#     cv2.line(frame, (w // 2, h), (w // 2, h - 30), (255, 0, 0), 2)     # קו אמצע
#     cv2.circle(frame, (cx, int(h * 0.8)), 8, (0, 255, 0), -1)
# # מיקום קו מזוהה
#     cv2.putText(frame, f"Error: {error}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     cv2.putText(frame, f"Correction: {correction}", (10, 60),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

#     cv2.imshow("Camera + Path Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
