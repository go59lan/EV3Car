import cv2
import socket
import numpy as np


EV3_IP = "10.42.0.3"
EV3_PORT = 5000


class RobotController:

    def __init__(self):

        # -----------------------------
        # Camera
        # -----------------------------

        self.cap = None
        self.latest_frame = None

        # -----------------------------
        # Robot state
        # -----------------------------

        self.running = False

        # -----------------------------
        # UDP connection to EV3
        # -----------------------------

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # -----------------------------
        # P-control
        # -----------------------------

        self.Kp = 0.6

        # -----------------------------
        # Smoothing for the detected
        # road-center (cx). Damps
        # frame-to-frame jitter from
        # noisy edge detection.
        # -----------------------------

        self.cx_filtered = None
        self.smoothing = 0.5  # higher = less smoothing

        # -----------------------------
        # Driving speed
        # -----------------------------

        self.speed = 15


    def run(self):

        self.running = True

        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Could not open camera")
            self.running = False
            return
        cx = None
        while self.running:

            # ==========================================
            # 1. Capture frame
            # ==========================================

            ret, frame = self.cap.read()

            if not ret:
                print("No camera frame")
                continue

            h, w = frame.shape[:2]


            # ==========================================
            # 2. Detect road edges
            # ==========================================

            left_line, right_line = self.detect_road(frame)


            # ==========================================
            # 3. Calculate road center
            # ==========================================

            
            if left_line is not None and right_line is not None:

                # Look further ahead (closer to the top of
                # the ROI) rather than right at the car.
                # A near-field target makes the error
                # collapse as soon as the car starts
                # turning its nose into a curve, even
                # though the turn isn't finished yet -
                # causing the correction to snap back too
                # early on sharp curves like the U-turn.

                y_target = int(h * 0.65)

                left_x = self.x_at_y(left_line, y_target)
                right_x = self.x_at_y(right_line, y_target)

                if left_x is not None and right_x is not None:

                    cx = int((left_x + right_x) / 2)

                    # Smooth cx to damp frame-to-frame
                    # jitter before it reaches the
                    # controller.

                    if self.cx_filtered is None:
                        self.cx_filtered = cx
                    else:
                        self.cx_filtered = int(
                            self.smoothing * cx
                            + (1 - self.smoothing) * self.cx_filtered
                        )

                    cx = self.cx_filtered

                    # Draw road center
                    cv2.circle(
                        frame,
                        (cx, y_target),
                        10,
                        (255, 0, 0),
                        -1
                    )

                    # Draw the two detected road edges
                    self.draw_line(
                        frame,
                        left_line,
                        y_target,
                        (0, 255, 0)
                    )

                    self.draw_line(
                        frame,
                        right_line,
                        y_target,
                        (0, 255, 0)
                    )


            # ==========================================
            # 4. P-control / navigation
            # ==========================================

            camera_center = w // 2

            # Camera center line
            cv2.line(
                frame,
                (camera_center, h),
                (camera_center, int(h * 0.55)),
                (255, 0, 0),
                2
            )


            if cx is not None:

                # --------------------------------------
                # Error
                # --------------------------------------

                error = camera_center - cx

                # --------------------------------------
                # P-control
                # --------------------------------------

                correction = int(self.Kp * error)

                # Limit steering correction
                correction = max(
                    min(correction, 130),
                    -130
                )

                # --------------------------------------
                # Send command to EV3
                # --------------------------------------

                msg = f"{correction},{self.speed}".encode()

                self.sock.sendto(
                    msg,
                    (EV3_IP, EV3_PORT)
                )

                print(
                    f"Sending to EV3: "
                    f"cx={cx}, "
                    f"error={error}, "
                    f"correction={correction}, "
                    f"speed={self.speed}"
                )

                # --------------------------------------
                # Debug text
                # --------------------------------------

                cv2.putText(
                    frame,
                    f"Road center: {cx}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Error: {error}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Correction: {correction}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            else:

                # ======================================
                # No road detected
                # ======================================

                cv2.putText(
                    frame,
                    "ROAD NOT DETECTED",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                # IMPORTANT:
                # Do not keep driving blindly if we
                # cannot see both road edges.

                self.sock.sendto(
                    b"0,0",
                    (EV3_IP, EV3_PORT)
                )

                # Road lost - drop the stale smoothed
                # estimate so it doesn't bias the first
                # reading once the road is found again.
                self.cx_filtered = None
                cx = None


            # ==========================================
            # 5. Encode frame for Flask
            # ==========================================

            success, jpeg = cv2.imencode(
                ".jpg",
                frame
            )

            if success:
                self.latest_frame = jpeg.tobytes()


        # ==============================================
        # 6. Cleanup
        # ==============================================

        self.stop()


    # ==================================================
    # ROAD DETECTION
    # ==================================================

    def detect_road(self, frame):

        h, w = frame.shape[:2]

        # ----------------------------------------------
        # Grayscale
        # ----------------------------------------------

        # gray = cv2.cvtColor(
        #     frame,
        #     cv2.COLOR_BGR2GRAY
        # )
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([40, 255, 255])

        yellow_mask = cv2.inRange(
            hsv,
            lower_yellow,
            upper_yellow
        )

        kernel = np.ones((5, 5), np.uint8)

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            kernel
        )


        # ----------------------------------------------
        # Region Of Interest
        #
        # Only look at the lower part of the image.
        #
        #        camera
        #    ______________
        #   |              |
        #   |              |
        #   |              |
        #   |    ______    |
        #   |   /      \   |
        #   |  /        \  |
        #   |_/__________\_|
        # ----------------------------------------------

        mask = np.zeros_like(yellow_mask)

        polygon = np.array([
            [
                (0, h),
                (w, h),
                (int(w), int(h * 0.55)),
                (int(0), int(h * 0.55))
            ]
        ], dtype=np.int32)

        cv2.fillPoly(
            mask,
            polygon,
            255
        )

        roi = cv2.bitwise_and(
            yellow_mask,
            mask
        )

        # ----------------------------------------------
        # Hough line detection
        # ----------------------------------------------

        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=30,
            maxLineGap=20
        )

        if lines is None:
            return None, None


        left_lines = []
        right_lines = []


        # ----------------------------------------------
        # Separate left and right lines
        # ----------------------------------------------

        for line in lines:

            x1, y1, x2, y2 = line[0]

            # Avoid division by zero
            if x2 == x1:
                continue

            slope = (
                (y2 - y1)
                / (x2 - x1)
            )

            # Ignore almost-horizontal lines
            if abs(slope) < 0.5:
                continue

            # In image coordinates:
            #
            # Left road edge generally has
            # negative slope.
            #
            # Right road edge generally has
            # positive slope.

            if slope < 0:
                left_lines.append(
                    (x1, y1, x2, y2)
                )

            else:
                right_lines.append(
                    (x1, y1, x2, y2)
                )


        # ----------------------------------------------
        # Keep only the outermost lines on each side.
        #
        # Obstacles in the middle of the road (like a
        # median island) also have yellow borders, and
        # those inner edges can slip into the same slope
        # bucket as the true road edge. Averaging with
        # them pulls the estimated edge toward the
        # island. Since the island's edges are always
        # closer to center than the real track boundary,
        # keeping only the lines nearest the outer extreme
        # filters the island out.
        # ----------------------------------------------

        left_line = self.select_outer_line(
            left_lines,
            "left"
        )

        if left_line is None:
            left_line = (0,0,0,h)

        right_line = self.select_outer_line(
            right_lines,
            "right"
        )
        if right_line is None:
            right_line = (w,0,w,h)
        return left_line, right_line


    # ==================================================
    # Keep only the lines nearest the outer edge of the
    # road on the given side, then average those.
    # ==================================================

    def select_outer_line(self, lines, side):

        if not lines:
            return None

        def avg_x(line):
            return (line[0] + line[2]) / 2

        if side == "left":
            extreme_x = min(avg_x(l) for l in lines)
        else:
            extreme_x = max(avg_x(l) for l in lines)

        tolerance = 40  # pixels

        outer_lines = [
            l for l in lines
            if abs(avg_x(l) - extreme_x) <= tolerance
        ]

        return self.average_line(outer_lines)


    # ==================================================
    # Average several detected lines
    # ==================================================

    def average_line(self, lines):

        if not lines:
            return None

        x1 = []
        y1 = []
        x2 = []
        y2 = []

        for line in lines:

            x1.append(line[0])
            y1.append(line[1])
            x2.append(line[2])
            y2.append(line[3])

        return (
            int(np.mean(x1)),
            int(np.mean(y1)),
            int(np.mean(x2)),
            int(np.mean(y2))
        )


    # ==================================================
    # Find X coordinate of a line at a given Y
    # ==================================================

    def x_at_y(self, line, y):

        if line is None:
            return None

        x1, y1, x2, y2 = line

        if y2 == y1:
            return None

        # Equation of a line:
        #
        # x = x1 + (y-y1) * (x2-x1)/(y2-y1)

        x = (
            x1
            + (y - y1)
            * (x2 - x1)
            / (y2 - y1)
        )

        return int(x)


    # ==================================================
    # Draw a detected line
    # ==================================================

    def draw_line(
        self,
        frame,
        line,
        y_target,
        color
    ):

        if line is None:
            return

        x1, y1, x2, y2 = line

        x_target = self.x_at_y(
            line,
            y_target
        )

        if x_target is None:
            return

        cv2.line(
            frame,
            (x1, y1),
            (x_target, y_target),
            color,
            3
        )


    # ==================================================
    # Stop robot
    # ==================================================

    def stop(self):

        self.running = False

        # Tell EV3 to stop
        try:
            self.sock.sendto(
                b"0,0",
                (EV3_IP, EV3_PORT)
            )
        except Exception:
            pass

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()


    # ==================================================
    # Get latest frame for Flask
    # ==================================================

    def get_frame(self):

        return self.latest_frame
