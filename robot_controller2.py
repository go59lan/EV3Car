import time
import cv2
import socket
import numpy as np


EV3_IP = "10.42.0.3"
EV3_PORT = 5000


class RobotController:
    """Road follower for a gray road with yellow edges and a dashed white center."""

    def __init__(self):
        self.cap = None
        self.latest_frame = None
        self.running = False

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # P control is enough once the image measurement is stable.
        self.Kp = 0.6

        # Center smoothing and steering rate limiting prevent one bad frame
        # from commanding a sudden full-lock turn.
        self.cx_filtered = None
        self.center_new_weight = 0.35
        self.last_correction = 0
        self.max_steering_step = 8

        self.speed = 10
        self.turn_speed = 7
        self.recovery_speed = 5
        self.recovery_seconds = 0.35
        self.lost_since = None

        # Geometry learned while both yellow borders are visible.
        self.lane_width = None
        self.last_left_x = None
        self.last_right_x = None

        # White-line history is kept separately from the general road center.
        # A yellow midpoint must not silently become a "previous white line".
        self.last_white_x = None
        self.white_missing_frames = 0
        self.last_trusted_center = None
        self.center_missing_frames = 0

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Could not open camera")
            self.running = False
            return

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("No camera frame")
                    continue

                h, w = frame.shape[:2]
                y_target = int(h * 0.65)

                # Everything is recalculated on each frame. Never reuse an old
                # cx as if it were a new camera measurement.
                cx = None
                source = "none"

                left_line, right_line, yellow_mask = self.detect_road(frame)
                yellow_center, yellow_mode = self.center_from_yellow(
                    left_line, right_line, y_target, w
                )

                center_line, white_mask = self.detect_center_line(
                    frame=frame,
                    left_line=left_line,
                    right_line=right_line,
                    expected_center=yellow_center,
                    y_target=y_target,
                )
                white_x = self.x_at_y(center_line, y_target)

                if white_x is not None:
                    self.last_white_x = white_x
                    self.white_missing_frames = 0
                else:
                    self.white_missing_frames += 1
                    if self.white_missing_frames > 20:
                        self.last_white_x = None

                # A white line is used only after detect_center_line has passed
                # the yellow-corridor, continuity, and dashed-pattern checks.
                if white_x is not None:
                    cx = float(white_x)
                    source = "white dash"
                elif yellow_center is not None:
                    cx = float(yellow_center)
                    source = "yellow " + yellow_mode

                if cx is not None:
                    self.lost_since = None
                    self.center_missing_frames = 0
                    self.last_trusted_center = cx

                    if self.cx_filtered is None:
                        self.cx_filtered = cx
                    else:
                        a = self.center_new_weight
                        self.cx_filtered = a * cx + (1.0 - a) * self.cx_filtered

                    cx = self.cx_filtered
                    camera_center = w / 2.0
                    error = camera_center - cx

                    desired = int(np.clip(self.Kp * error, -100, 100))
                    correction = self.limit_steering_change(desired)

                    speed = self.speed
                    if abs(correction) >= 45 or yellow_mode != "both":
                        speed = self.turn_speed

                    self.send_command(correction, speed)
                    self.draw_tracking(
                        frame, cx, y_target, error, correction, speed, source
                    )
                else:
                    self.center_missing_frames += 1
                    if self.center_missing_frames > 20:
                        self.last_trusted_center = None
                        self.cx_filtered = None
                    self.recover_or_stop(frame)

                camera_center = w // 2
                cv2.line(
                    frame,
                    (camera_center, h - 1),
                    (camera_center, int(h * 0.52)),
                    (255, 0, 0),
                    2,
                )

                if left_line is not None:
                    self.draw_line(frame, left_line, y_target, (0, 255, 0))
                if right_line is not None:
                    self.draw_line(frame, right_line, y_target, (0, 255, 0))
                if center_line is not None:
                    self.draw_line(frame, center_line, y_target, (255, 0, 255))

                self.draw_mask_previews(frame, yellow_mask, white_mask)

                success, jpeg = cv2.imencode(".jpg", frame)
                if success:
                    self.latest_frame = jpeg.tobytes()
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # Yellow road-border detection
    # ------------------------------------------------------------------

    def detect_road(self, frame):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(
            hsv,
            np.array([14, 70, 65], dtype=np.uint8),
            np.array([42, 255, 255], dtype=np.uint8),
        )

        # A small kernel preserves thin yellow fragments at the frame edges.
        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(
            yellow_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        yellow_mask = cv2.morphologyEx(
            yellow_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        roi_top = int(h * 0.50)
        roi = self.apply_roi(yellow_mask, roi_top)

        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=16,
            minLineLength=14,
            maxLineGap=35,
        )

        if lines is None:
            return None, None, yellow_mask

        y_reference = int(h * 0.70)
        candidates = []

        for detected in lines[:, 0]:
            segment = self.normalize_segment(tuple(map(int, detected)))
            x1, y1, x2, y2 = segment
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))

            # This accepts vertical lines. The original `x2 == x1: continue`
            # discarded exactly the lines seen on a straight road.
            if length < 14 or abs(dy) < 8 or abs(dy) < 0.35 * abs(dx):
                continue

            x_reference = self.x_at_y(segment, y_reference)
            if x_reference is None or not (-w <= x_reference <= 2 * w):
                continue

            candidates.append((x_reference, length, segment))

        if not candidates:
            return None, None, yellow_mask

        # Merge repeated Hough segments from the same painted line, but keep
        # nearby lines from different road sections as separate candidates.
        clusters = self.cluster_segments(candidates, max(14, int(w * 0.04)))

        # A different part of the road can also be visible in the camera,
        # especially around a U-turn. Do not select the outermost yellow
        # clusters. Split them at the image center and keep only the innermost
        # cluster on each side:
        #
        #   left side  -> greatest X (closest to the middle)
        #   right side -> smallest X (closest to the middle)
        #
        # If all clusters are on the same side, this deliberately returns only
        # one line. For example, with two lines on the left, the line closer to
        # the middle is kept and the real right border is treated as out of view.

        left_cluster, right_cluster = self.select_center_side_clusters(
            clusters, w
        )

        left_line = self.fit_cluster(left_cluster, roi_top, h - 1)
        right_line = self.fit_cluster(right_cluster, roi_top, h - 1)

        return left_line, right_line, yellow_mask

    def center_from_yellow(self, left_line, right_line, y_target, image_width):
        left_x = self.x_at_y(left_line, y_target)
        right_x = self.x_at_y(right_line, y_target)

        if left_x is not None and right_x is not None:
            if right_x < left_x:
                left_x, right_x = right_x, left_x

            measured_width = right_x - left_x
            if measured_width >= image_width * 0.20:
                measured_width = float(
                    np.clip(
                        measured_width,
                        image_width * 0.25,
                        image_width * 1.60,
                    )
                )
                if self.lane_width is None:
                    self.lane_width = measured_width
                else:
                    self.lane_width = 0.85 * self.lane_width + 0.15 * measured_width

                self.last_left_x = left_x
                self.last_right_x = right_x
                return (left_x + right_x) / 2.0, "both"

        # If only one border is visible, infer the center from the road width
        # learned on previous frames. Never invent the missing line at x=0/w.
        if self.lane_width is not None:
            if left_x is not None:
                self.last_left_x = left_x
                return left_x + self.lane_width / 2.0, "left only"
            if right_x is not None:
                self.last_right_x = right_x
                return right_x - self.lane_width / 2.0, "right only"

        return None, "lost"

    # ------------------------------------------------------------------
    # Dashed white center-line detection
    # ------------------------------------------------------------------

    def detect_center_line(
        self,
        frame,
        left_line=None,
        right_line=None,
        expected_center=None,
        y_target=None,
    ):
        h, w = frame.shape[:2]
        if y_target is None:
            y_target = int(h * 0.65)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 175], dtype=np.uint8),
            np.array([179, 65, 255], dtype=np.uint8),
        )

        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(
            white_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        white_mask = cv2.morphologyEx(
            white_mask, cv2.MORPH_CLOSE, kernel, iterations=1
        )

        roi_top = int(h * 0.50)
        roi = self.apply_roi(white_mask, roi_top)

        # Keep dashes separate. A large maxLineGap can turn a dashed line and a
        # solid mat border into similarly long Hough lines.
        lines = cv2.HoughLinesP(
            roi,
            rho=1,
            theta=np.pi / 180,
            threshold=12,
            minLineLength=10,
            maxLineGap=22,
        )

        if lines is None:
            return None, white_mask

        reference = expected_center
        if reference is None:
            reference = self.last_trusted_center
        if reference is None:
            reference = self.last_white_x
        if reference is None:
            # Conservative first acquisition: a center dash should begin near
            # the camera center. An outside mat border should not.
            reference = w / 2.0
            reference_limit = w * 0.20
        else:
            reference_limit = max(45.0, w * 0.15)

        accepted = []

        for detected in lines[:, 0]:
            segment = self.normalize_segment(tuple(map(int, detected)))
            x1, y1, x2, y2 = segment
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))

            # Reject horizontal mat markings but retain vertical center dashes.
            if length < 10 or abs(dy) < 7 or abs(dy) < 0.35 * abs(dx):
                continue

            candidate_x = self.x_at_y(segment, y_target)
            if candidate_x is None or not (0 <= candidate_x < w):
                continue

            reference_error = abs(candidate_x - reference)
            if reference_error > reference_limit:
                continue

            # Hard temporal gate. The previous version picked the nearest line
            # even when every candidate was far away, which let the map border
            # hijack the detector in a single frame.
            temporal_error = 0.0
            if self.last_white_x is not None:
                temporal_error = abs(candidate_x - self.last_white_x)
                if temporal_error > max(55.0, w * 0.14):
                    continue

            center_error = self.center_band_error(
                segment, left_line, right_line, w
            )
            if center_error is None and left_line is not None and right_line is not None:
                # Both yellow borders exist but this white candidate could not
                # be proven to lie between them at matching Y-coordinates.
                continue
            if center_error is not None and center_error > 0.32:
                # 0 means the exact yellow midpoint; 0.5 means a yellow edge.
                # The dashed line must stay in the central part of the road.
                continue

            occupancy, longest_run_ratio = self.line_pattern(
                white_mask, segment, roi_top, h - 1
            )
            if occupancy > 0.80 and longest_run_ratio > 0.65:
                # A solid white map border remains white for most of the ROI.
                # A dashed center line necessarily contains visible gaps.
                continue

            geometry_penalty = 0.0 if center_error is None else center_error * w
            solid_penalty = max(0.0, occupancy - 0.55) * 80.0
            score = (
                reference_error
                + 0.35 * temporal_error
                + geometry_penalty
                + solid_penalty
                - min(length, 80.0) * 0.05
            )
            accepted.append((score, candidate_x, length, segment))

        if not accepted:
            return None, white_mask

        accepted.sort(key=lambda item: item[0])
        best_x = accepted[0][1]

        # Combine only segments belonging to the winning dash trajectory.
        matching = [
            (item[1], item[2], item[3])
            for item in accepted
            if abs(item[1] - best_x) <= max(18, int(w * 0.045))
        ]
        center_line = self.fit_cluster(matching, roi_top, h - 1)

        final_x = self.x_at_y(center_line, y_target)
        if final_x is None or abs(final_x - reference) > reference_limit:
            return None, white_mask

        return center_line, white_mask

    def center_band_error(self, candidate, left_line, right_line, image_width):
        """Compare white and yellow geometry at identical image heights."""
        if left_line is None or right_line is None:
            return None

        _, y1, _, y2 = candidate
        sample_y_values = (y1, (y1 + y2) // 2, y2)
        normalized_errors = []

        for y in sample_y_values:
            white_x = self.x_at_y(candidate, y)
            left_x = self.x_at_y(left_line, y)
            right_x = self.x_at_y(right_line, y)
            if white_x is None or left_x is None or right_x is None:
                continue

            road_min = min(left_x, right_x)
            road_max = max(left_x, right_x)
            road_width = road_max - road_min
            if road_width < image_width * 0.15:
                continue

            road_center = (road_min + road_max) / 2.0
            normalized_errors.append(abs(white_x - road_center) / road_width)

        if not normalized_errors:
            return None
        return float(np.mean(normalized_errors))

    def line_pattern(self, mask, line, y_start, y_end):
        """Measure whether a proposed white line is dashed or continuous."""
        h, w = mask.shape[:2]
        samples = []

        for y in range(max(0, y_start), min(h - 1, y_end) + 1, 2):
            x = self.x_at_y(line, y)
            if x is None or x < 0 or x >= w:
                continue
            x1 = max(0, x - 3)
            x2 = min(w, x + 4)
            samples.append(bool(np.any(mask[y, x1:x2] > 0)))

        if not samples:
            return 0.0, 0.0

        occupied = sum(samples)
        longest_run = 0
        current_run = 0
        for is_white in samples:
            if is_white:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 0

        return occupied / len(samples), longest_run / len(samples)

    # ------------------------------------------------------------------
    # Shared image geometry
    # ------------------------------------------------------------------

    @staticmethod
    def apply_roi(mask, roi_top):
        roi_mask = np.zeros_like(mask)
        roi_mask[roi_top:, :] = 255
        return cv2.bitwise_and(mask, roi_mask)

    @staticmethod
    def normalize_segment(line):
        x1, y1, x2, y2 = line
        if y1 <= y2:
            return x1, y1, x2, y2
        return x2, y2, x1, y1

    @staticmethod
    def cluster_x(cluster):
        total_weight = sum(item[1] for item in cluster)
        return sum(item[0] * item[1] for item in cluster) / total_weight

    @staticmethod
    def cluster_segments(candidates, gap):
        candidates = sorted(candidates, key=lambda item: item[0])
        clusters = []
        for candidate in candidates:
            if not clusters or candidate[0] - clusters[-1][-1][0] > gap:
                clusters.append([candidate])
            else:
                clusters[-1].append(candidate)
        return clusters

    def select_center_side_clusters(self, clusters, image_width):
        """Keep at most one yellow cluster on each side of image center."""
        image_center = image_width / 2.0

        left_clusters = [
            cluster
            for cluster in clusters
            if self.cluster_x(cluster) < image_center
        ]
        right_clusters = [
            cluster
            for cluster in clusters
            if self.cluster_x(cluster) >= image_center
        ]

        # On the left, a larger X is closer to the middle. On the right, a
        # smaller X is closer to the middle.
        left_cluster = (
            max(left_clusters, key=self.cluster_x)
            if left_clusters
            else None
        )
        right_cluster = (
            min(right_clusters, key=self.cluster_x)
            if right_clusters
            else None
        )

        return left_cluster, right_cluster

    @staticmethod
    def fit_cluster(cluster, y_top, y_bottom):
        if not cluster:
            return None

        points = []
        for _, _, (x1, y1, x2, y2) in cluster:
            points.extend(((x1, y1), (x2, y2)))

        points = np.asarray(points, dtype=np.float32)
        fit = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        vx, vy, x0, y0 = [float(value) for value in fit]

        if abs(vy) < 1e-6:
            return None

        x_top = x0 + (y_top - y0) * vx / vy
        x_bottom = x0 + (y_bottom - y0) * vx / vy
        return int(x_top), int(y_top), int(x_bottom), int(y_bottom)

    @staticmethod
    def x_at_y(line, y):
        if line is None:
            return None

        x1, y1, x2, y2 = line
        if y2 == y1:
            return None

        return int(x1 + (y - y1) * (x2 - x1) / (y2 - y1))

    # ------------------------------------------------------------------
    # Driving and debug display
    # ------------------------------------------------------------------

    def limit_steering_change(self, desired):
        low = self.last_correction - self.max_steering_step
        high = self.last_correction + self.max_steering_step
        correction = int(np.clip(desired, low, high))
        correction = int(np.clip(correction, -100, 100))
        self.last_correction = correction
        return correction

    def recover_or_stop(self, frame):
        now = time.monotonic()
        if self.lost_since is None:
            self.lost_since = now

        elapsed = now - self.lost_since
        if elapsed <= self.recovery_seconds and self.last_correction != 0:
            # Preserve the turn briefly instead of steering straight at the
            # exact moment the curve moves out of the narrow camera view.
            self.send_command(self.last_correction, self.recovery_speed)
            text = "RECOVERING ROAD"
            color = (0, 165, 255)
        else:
            self.send_command(0, 0)
            self.last_correction = 0
            text = "ROAD LOST - STOPPED"
            color = (0, 0, 255)

        cv2.putText(
            frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    def send_command(self, correction, speed):
        message = "{},{}".format(int(correction), int(speed)).encode()
        self.sock.sendto(message, (EV3_IP, EV3_PORT))
        print(
            "Sending to EV3: correction={}, speed={}".format(
                int(correction), int(speed)
            )
        )

    @staticmethod
    def draw_tracking(frame, cx, y_target, error, correction, speed, source):
        h, w = frame.shape[:2]
        draw_x = int(np.clip(cx, 0, w - 1))
        cv2.circle(frame, (draw_x, y_target), 9, (255, 0, 0), -1)

        labels = (
            "Road center: {:.0f}".format(cx),
            "Error: {:.0f}".format(error),
            "Correction: {}  Speed: {}".format(correction, speed),
            "Source: {}".format(source),
        )
        for index, label in enumerate(labels):
            cv2.putText(
                frame,
                label,
                (10, 28 + index * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )

    def draw_line(self, frame, line, y_target, color):
        if line is None:
            return

        h, w = frame.shape[:2]
        x1, y1, _, _ = line
        x_target = self.x_at_y(line, y_target)
        if x_target is None:
            return

        cv2.line(
            frame,
            (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1))),
            (int(np.clip(x_target, 0, w - 1)), y_target),
            color,
            3,
        )

    @staticmethod
    def draw_mask_previews(frame, yellow_mask, white_mask):
        h, w = frame.shape[:2]
        preview_w = max(1, w // 5)
        preview_h = max(1, h // 5)

        yellow = cv2.resize(yellow_mask, (preview_w, preview_h))
        yellow = cv2.cvtColor(yellow, cv2.COLOR_GRAY2BGR)
        yellow[:, :, 0] = 0

        white = cv2.resize(white_mask, (preview_w, preview_h))
        white = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)

        x1 = w - preview_w
        frame[0:preview_h, x1:w] = yellow
        if 2 * preview_h <= h:
            frame[preview_h:2 * preview_h, x1:w] = white

    def stop(self):
        self.running = False
        try:
            self.send_command(0, 0)
        except OSError:
            pass

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        cv2.destroyAllWindows()

    def get_frame(self):
        return self.latest_frame