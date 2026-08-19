import time
import cv2
import socket
import numpy as np


EV3_IP = "10.42.0.3"
EV3_PORT = 5000


class RobotController:
    """Follow yellow road borders with connected bottom-up scanlines."""

    def __init__(self):
        self.cap = None
        self.latest_frame = None
        self.running = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Steering control. The image measurement is smoothed and the command
        # is rate-limited, so a single bad frame cannot produce full lock.
        self.Kp = 0.6
        self.target_filtered = None
        self.last_correction = 0
        self.max_steering_step = 8
        self.turn_steering_step = 14

        self.speed = 10
        self.turn_speed = 6
        self.recovery_speed = 5
        self.recovery_seconds = 0.35
        self.lost_since = None
        self.missing_frames = 0

        # Width is learned independently at the different scan heights. That
        # matters because perspective makes the road narrower near the top.
        self.width_profile = {}
        self.near_lane_width = None

        self.last_trusted_center = None
        self.last_left_near_x = None
        self.last_right_near_x = None

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
                detection = self.detect_road(frame)
                navigation = self.calculate_navigation(detection, w, h)

                if navigation is not None:
                    self.lost_since = None
                    self.missing_frames = 0

                    target_x = navigation["target_x"]
                    turn_strength = navigation["turn_strength"]

                    # React faster when a connected horizontal yellow segment
                    # proves that the border is turning out of the frame.
                    new_weight = 0.65 if turn_strength > 0 else 0.35
                    if self.target_filtered is None:
                        self.target_filtered = target_x
                    else:
                        self.target_filtered = (
                            new_weight * target_x
                            + (1.0 - new_weight) * self.target_filtered
                        )

                    camera_center = w / 2.0
                    error = camera_center - self.target_filtered
                    desired = int(np.clip(self.Kp * error, -100, 100))

                    maximum_step = (
                        self.turn_steering_step
                        if turn_strength > 0
                        else self.max_steering_step
                    )
                    correction = self.limit_steering_change(
                        desired, maximum_step
                    )

                    speed = self.speed
                    if (
                        turn_strength > 0
                        or abs(correction) >= 45
                        or navigation["mode"] != "both"
                    ):
                        speed = self.turn_speed

                    self.last_trusted_center = self.target_filtered
                    self.send_command(correction, speed)
                    self.draw_status(
                        frame,
                        self.target_filtered,
                        error,
                        correction,
                        speed,
                        navigation,
                    )
                else:
                    self.missing_frames += 1
                    if self.missing_frames > 20:
                        self.target_filtered = None
                        self.last_trusted_center = None
                    self.recover_or_stop(frame)

                self.draw_detection(frame, detection)

                camera_center = w // 2
                cv2.line(
                    frame,
                    (camera_center, h - 1),
                    (camera_center, int(h * 0.44)),
                    (255, 0, 0),
                    2,
                )

                success, jpeg = cv2.imencode(".jpg", frame)
                if success:
                    self.latest_frame = jpeg.tobytes()
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # Yellow mask and connected scanline paths
    # ------------------------------------------------------------------

    def make_yellow_mask(self, frame):
        h, _ = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Tuned from the measured RGB yellow samples. Raising the minimum hue
        # rejects the brown sample that was previously classified as yellow.
        yellow_mask = cv2.inRange(
            hsv,
            np.array([28, 75, 135], dtype=np.uint8),
            np.array([42, 180, 255], dtype=np.uint8),
        )

        kernel = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(
            yellow_mask, cv2.MORPH_OPEN, kernel, iterations=1
        )
        yellow_mask = cv2.morphologyEx(
            yellow_mask, cv2.MORPH_CLOSE, kernel, iterations=2
        )

        # Look from 44% of the image height down to the bottom. Unlike Hough
        # filtering, no orientation is rejected: vertical, curved, diagonal,
        # and fully horizontal yellow markings all remain in this mask.
        roi_top = int(h * 0.44)
        yellow_mask[:roi_top, :] = 0
        return yellow_mask, roi_top

    def detect_road(self, frame):
        h, w = frame.shape[:2]
        yellow_mask, roi_top = self.make_yellow_mask(frame)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            yellow_mask, connectivity=8
        )

        minimum_area = max(18, int(w * h * 0.00007))
        minimum_span = max(10, int(min(w, h) * 0.025))
        valid_labels = []

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])

            # Accept a component if it is long in either direction. This is
            # what allows a fully horizontal border at a sharp turn to survive.
            if area >= minimum_area and max(
                component_width, component_height
            ) >= minimum_span:
                valid_labels.append(label)

        component_runs = self.extract_component_runs(
            labels, stats, valid_labels
        )

        # A component must reach the lower 28% of the frame to seed a road
        # border. Yellow from another road visible only in the distance is not
        # allowed to start a path.
        seed_zone_top = int(h * 0.72)
        paths = []

        for label in valid_labels:
            runs_by_y = component_runs.get(label)
            if not runs_by_y:
                continue

            bottom_y = max(runs_by_y)
            if bottom_y < seed_zone_top:
                continue

            path = self.trace_component(label, runs_by_y, w)
            if path is not None:
                paths.append(path)

        left_path, right_path = self.select_current_borders(paths, w)

        scan_step = max(8, int(h * 0.025))
        scan_y_values = list(range(h - 6, roi_top - 1, -scan_step))
        centers = self.build_center_samples(
            left_path,
            right_path,
            scan_y_values,
            w,
        )

        turn_direction, turn_strength = self.horizontal_turn_hint(
            left_path, right_path, w
        )

        return {
            "yellow_mask": yellow_mask,
            "roi_top": roi_top,
            "scan_y_values": scan_y_values,
            "candidate_paths": paths,
            "left_path": left_path,
            "right_path": right_path,
            "centers": centers,
            "turn_direction": turn_direction,
            "turn_strength": turn_strength,
        }

    @staticmethod
    def extract_component_runs(labels, stats, valid_labels):
        """Return every contiguous X-run for every component and image row."""
        component_runs = {}

        for label in valid_labels:
            x0 = int(stats[label, cv2.CC_STAT_LEFT])
            y0 = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            runs_by_y = {}

            for y in range(y0, y0 + height):
                xs = np.flatnonzero(labels[y, x0:x0 + width] == label)
                if xs.size == 0:
                    continue

                xs = xs + x0
                split_after = np.flatnonzero(np.diff(xs) > 1)
                starts = np.concatenate(([0], split_after + 1))
                ends = np.concatenate((split_after, [xs.size - 1]))

                runs_by_y[y] = [
                    (int(xs[start]), int(xs[end]))
                    for start, end in zip(starts, ends)
                ]

            component_runs[label] = runs_by_y

        return component_runs

    def trace_component(self, label, runs_by_y, image_width):
        """Trace one connected yellow component from its bottom toward the top."""
        if not runs_by_y:
            return None

        y_values = sorted(runs_by_y, reverse=True)
        bottom_y = y_values[0]

        # At the first row choose the run nearest the image center. Branches
        # farther from the current road are ignored.
        seed_run = min(
            runs_by_y[bottom_y],
            key=lambda run: abs(self.run_center(run) - image_width / 2.0),
        )
        anchor_x = self.run_center(seed_run)

        horizontal_minimum = max(45, int(image_width * 0.10))
        active_horizontal_direction = 0
        points = []
        points_by_y = {}
        horizontal_segments = []

        for y in y_values:
            row_runs = runs_by_y[y]
            run = min(
                row_runs,
                key=lambda candidate: self.distance_to_run(anchor_x, candidate),
            )

            x1, x2 = run
            run_width = x2 - x1 + 1

            if run_width >= horizontal_minimum:
                if active_horizontal_direction == 0:
                    distance_left = abs(anchor_x - x1)
                    distance_right = abs(x2 - anchor_x)

                    # If the path entered near one endpoint, the far endpoint
                    # shows the direction of the turn. If it entered near the
                    # exact middle, the direction is ambiguous and no hint is
                    # generated from that row.
                    difference = abs(distance_right - distance_left)
                    if difference >= horizontal_minimum * 0.20:
                        active_horizontal_direction = (
                            1 if distance_right > distance_left else -1
                        )

                if active_horizontal_direction > 0:
                    chosen_x = float(x2)
                elif active_horizontal_direction < 0:
                    chosen_x = float(x1)
                else:
                    chosen_x = float(np.clip(anchor_x, x1, x2))

                # Store even an ambiguous standalone horizontal line. After the
                # path is assigned to the left or right border, its position in
                # the previous frame can resolve which endpoint it entered.
                horizontal_segments.append(
                    {
                        "x1": x1,
                        "x2": x2,
                        "y": y,
                        "length": run_width,
                        "direction": active_horizontal_direction,
                    }
                )
            else:
                chosen_x = self.run_center(run)
                active_horizontal_direction = 0

            points.append((chosen_x, y))
            points_by_y[y] = chosen_x
            anchor_x = chosen_x

        # Median over the bottom 12 rows is more stable than one edge pixel.
        near_points = [x for x, y in points if y >= bottom_y - 12]
        near_x = float(np.median(near_points))

        return {
            "label": label,
            "bottom_y": bottom_y,
            "near_x": near_x,
            "points": points,
            "points_by_y": points_by_y,
            "horizontal_segments": horizontal_segments,
        }

    @staticmethod
    def run_center(run):
        return (run[0] + run[1]) / 2.0

    @staticmethod
    def distance_to_run(x, run):
        if run[0] <= x <= run[1]:
            return 0.0
        return min(abs(x - run[0]), abs(x - run[1]))

    def select_current_borders(self, paths, image_width):
        """Keep the bottom-connected path nearest center on each side."""
        image_center = image_width / 2.0
        previous_left_x = self.last_left_near_x
        previous_right_x = self.last_right_near_x
        left_candidates = [
            path for path in paths if path["near_x"] < image_center
        ]
        right_candidates = [
            path for path in paths if path["near_x"] >= image_center
        ]

        left_path = (
            max(left_candidates, key=lambda path: path["near_x"])
            if left_candidates
            else None
        )
        right_path = (
            min(right_candidates, key=lambda path: path["near_x"])
            if right_candidates
            else None
        )

        self.resolve_ambiguous_horizontal(left_path, previous_left_x)
        self.resolve_ambiguous_horizontal(right_path, previous_right_x)

        if left_path is not None:
            self.last_left_near_x = left_path["near_x"]
        if right_path is not None:
            self.last_right_near_x = right_path["near_x"]

        return left_path, right_path

    @staticmethod
    def resolve_ambiguous_horizontal(path, previous_x):
        """Use the preceding frame to orient a standalone horizontal border."""
        if path is None or previous_x is None:
            return

        for segment in path["horizontal_segments"]:
            if segment["direction"] != 0:
                continue

            distance_left = abs(previous_x - segment["x1"])
            distance_right = abs(segment["x2"] - previous_x)
            difference = abs(distance_right - distance_left)

            if difference >= segment["length"] * 0.20:
                segment["direction"] = (
                    1 if distance_right > distance_left else -1
                )

    # ------------------------------------------------------------------
    # Center path and turn calculation
    # ------------------------------------------------------------------

    def build_center_samples(
        self,
        left_path,
        right_path,
        scan_y_values,
        image_width,
    ):
        centers = []
        search_radius = max(4, int(len(scan_y_values) * 0.15))

        for level, y in enumerate(scan_y_values):
            left_x = self.path_x_at_y(left_path, y, search_radius)
            right_x = self.path_x_at_y(right_path, y, search_radius)
            center_x = None
            mode = "lost"

            if left_x is not None and right_x is not None:
                measured_width = right_x - left_x

                if measured_width >= image_width * 0.10:
                    old_width = self.width_profile.get(level)
                    if old_width is None:
                        learned_width = measured_width
                    else:
                        learned_width = 0.85 * old_width + 0.15 * measured_width
                    self.width_profile[level] = learned_width

                    if level <= 3:
                        if self.near_lane_width is None:
                            self.near_lane_width = measured_width
                        else:
                            self.near_lane_width = (
                                0.85 * self.near_lane_width
                                + 0.15 * measured_width
                            )

                    center_x = (left_x + right_x) / 2.0
                    mode = "both"

            if center_x is None:
                expected_width = self.width_for_level(level)

                if expected_width is not None and left_x is not None:
                    center_x = left_x + expected_width / 2.0
                    mode = "left only"
                elif expected_width is not None and right_x is not None:
                    center_x = right_x - expected_width / 2.0
                    mode = "right only"

            if center_x is not None:
                centers.append(
                    {
                        "x": float(center_x),
                        "y": y,
                        "level": level,
                        "mode": mode,
                    }
                )

        return centers

    def width_for_level(self, level):
        if level in self.width_profile:
            return self.width_profile[level]

        if self.width_profile:
            nearest_level = min(
                self.width_profile,
                key=lambda known: abs(known - level),
            )
            return self.width_profile[nearest_level]

        return self.near_lane_width

    @staticmethod
    def path_x_at_y(path, target_y, radius):
        if path is None:
            return None

        points_by_y = path["points_by_y"]
        if target_y in points_by_y:
            return points_by_y[target_y]

        nearest_y = None
        nearest_distance = radius + 1
        for y in points_by_y:
            distance = abs(y - target_y)
            if distance < nearest_distance:
                nearest_y = y
                nearest_distance = distance

        if nearest_y is None or nearest_distance > radius:
            return None
        return points_by_y[nearest_y]

    @staticmethod
    def horizontal_turn_hint(left_path, right_path, image_width):
        strongest_segments = []

        for path in (left_path, right_path):
            if path is None:
                continue
            directed_segments = [
                segment
                for segment in path["horizontal_segments"]
                if segment["direction"] != 0
            ]
            if not directed_segments:
                continue
            strongest_segments.append(
                max(
                    directed_segments,
                    key=lambda segment: segment["length"],
                )
            )

        if not strongest_segments:
            return 0, 0.0

        vote = sum(
            segment["direction"] * segment["length"]
            for segment in strongest_segments
        )

        # Opposing equally strong components are ambiguous, so do not force a
        # turn. Agreement or one clear connected segment creates a turn hint.
        if abs(vote) < image_width * 0.05:
            return 0, 0.0

        direction = 1 if vote > 0 else -1
        strength = min(1.0, abs(vote) / (image_width * 0.30))
        return direction, strength

    def calculate_navigation(self, detection, image_width, image_height):
        centers = detection["centers"]
        turn_direction = detection["turn_direction"]
        turn_strength = detection["turn_strength"]

        if centers:
            # scan_y_values run from bottom to top, so the first center is the
            # car's current position and the last is the farthest connected
            # look-ahead point.
            near = centers[0]
            far = centers[-1]
            target_x = near["x"] + 0.75 * (far["x"] - near["x"])
            mode = far["mode"]
        elif turn_direction != 0 and self.last_trusted_center is not None:
            # A horizontal border can remain visible for a moment after the
            # opposite border disappears. Continue from the last safe center.
            near = {
                "x": self.last_trusted_center,
                "y": int(image_height * 0.75),
            }
            far = near
            target_x = self.last_trusted_center
            mode = "horizontal only"
        else:
            return None

        if turn_direction != 0:
            target_x += (
                turn_direction
                * image_width
                * 0.20
                * turn_strength
            )

        target_x = float(np.clip(target_x, 0, image_width - 1))

        return {
            "target_x": target_x,
            "near": near,
            "far": far,
            "mode": mode,
            "turn_direction": turn_direction,
            "turn_strength": turn_strength,
        }

    # ------------------------------------------------------------------
    # Driving and debug view
    # ------------------------------------------------------------------

    def limit_steering_change(self, desired, maximum_step=None):
        if maximum_step is None:
            maximum_step = self.max_steering_step

        low = self.last_correction - maximum_step
        high = self.last_correction + maximum_step
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
            self.send_command(self.last_correction, self.recovery_speed)
            message = "RECOVERING ROAD"
            color = (0, 165, 255)
        else:
            self.send_command(0, 0)
            self.last_correction = 0
            message = "ROAD LOST - STOPPED"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            message,
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
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
    def draw_status(
        frame,
        target_x,
        error,
        correction,
        speed,
        navigation,
    ):
        direction_names = {-1: "LEFT", 0: "NONE", 1: "RIGHT"}
        labels = (
            "Target: {:.0f}".format(target_x),
            "Error: {:.0f}".format(error),
            "Correction: {}  Speed: {}".format(correction, speed),
            "Borders: {}".format(navigation["mode"]),
            "Horizontal turn: {} {:.2f}".format(
                direction_names[navigation["turn_direction"]],
                navigation["turn_strength"],
            ),
        )

        for index, label in enumerate(labels):
            cv2.putText(
                frame,
                label,
                (10, 27 + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
            )

    def draw_detection(self, frame, detection):
        h, w = frame.shape[:2]

        # Draw every second scanline to show the rows used for path sampling.
        for index, y in enumerate(detection["scan_y_values"]):
            if index % 2 == 0:
                cv2.line(frame, (0, y), (w - 1, y), (70, 70, 70), 1)

        self.draw_path(frame, detection["left_path"], (0, 255, 0))
        self.draw_path(frame, detection["right_path"], (0, 200, 255))

        center_points = [
            (int(center["x"]), int(center["y"]))
            for center in detection["centers"]
            if 0 <= center["x"] < w
        ]
        if len(center_points) >= 2:
            points = np.asarray(center_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [points], False, (255, 0, 0), 3)
        for point in center_points:
            cv2.circle(frame, point, 4, (255, 0, 0), -1)

        # Mask preview in the upper-right corner.
        preview_w = max(1, w // 5)
        preview_h = max(1, h // 5)
        preview = cv2.resize(
            detection["yellow_mask"], (preview_w, preview_h)
        )
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
        preview[:, :, 0] = 0
        frame[0:preview_h, w - preview_w:w] = preview

    @staticmethod
    def draw_path(frame, path, color):
        if path is None:
            return

        h, w = frame.shape[:2]
        # One point every four rows is enough for a smooth debug curve.
        visible = [
            (int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1)))
            for index, (x, y) in enumerate(path["points"])
            if index % 4 == 0
        ]

        if len(visible) >= 2:
            points = np.asarray(visible, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [points], False, color, 3)

        # Horizontal evidence is orange so it is easy to verify in the stream.
        for segment in path["horizontal_segments"]:
            cv2.line(
                frame,
                (segment["x1"], segment["y"]),
                (segment["x2"], segment["y"]),
                (0, 140, 255),
                2,
            )

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
