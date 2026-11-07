
import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

GOOD_ANGLE = 12.0
BAD_ANGLE = 18.0

GOOD_HEAD_RATIO = 0.12
BAD_HEAD_RATIO = 0.18

SMOOTHING_SECONDS = 0.5
BAD_CONFIRM_SECONDS = 1.5
GOOD_CONFIRM_SECONDS = 1.0

MIN_VISIBILITY = 0.5
MAX_FRAME_GAP = 0.25


# --------------------------------------------------
# 2. Landmark utilities
# --------------------------------------------------

def midpoint(point_a, point_b):
    """Calculate midpoint between two landmarks."""
    return np.array([
        (point_a.x + point_b.x) / 2,
        (point_a.y + point_b.y) / 2
    ])


def extract_features(landmarks):
    """
    Extract scale-normalized posture features
    from MediaPipe pose landmarks.
    """

    required = [
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        mp.solutions.pose.PoseLandmark.LEFT_HIP,
        mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        mp.solutions.pose.PoseLandmark.LEFT_EAR,
        mp.solutions.pose.PoseLandmark.RIGHT_EAR
    ]

    points = [landmarks[i.value] for i in required]

    if any(p.visibility < MIN_VISIBILITY for p in points):
        return None

    left_shoulder = landmarks[
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value
    ]
    right_shoulder = landmarks[
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value
    ]

    left_hip = landmarks[
        mp.solutions.pose.PoseLandmark.LEFT_HIP.value
    ]
    right_hip = landmarks[
        mp.solutions.pose.PoseLandmark.RIGHT_HIP.value
    ]

    left_ear = landmarks[
        mp.solutions.pose.PoseLandmark.LEFT_EAR.value
    ]
    right_ear = landmarks[
        mp.solutions.pose.PoseLandmark.RIGHT_EAR.value
    ]

    shoulder = midpoint(left_shoulder, right_shoulder)
    hip = midpoint(left_hip, right_hip)
    ear = midpoint(left_ear, right_ear)

    torso = shoulder - hip

    torso_length = np.linalg.norm(torso)

    if torso_length < 1e-6:
        return None

    # Torso angle relative to the vertical image axis.
    angle = math.degrees(
        math.atan2(abs(torso[0]), abs(torso[1]))
    )

    # Normalize head displacement by torso length.
    head_ratio = abs(ear[0] - shoulder[0]) / torso_length

    return {
        "torso_angle": float(angle),
        "head_ratio": float(head_ratio)
    }


# --------------------------------------------------
# 3. Temporal posture classifier
# --------------------------------------------------

class PostureClassifier:

    def __init__(self):
        self.history = deque()

        self.state = "good"

        self.bad_pending = 0.0
        self.good_pending = 0.0

        self.good_seconds = 0.0
        self.bad_seconds = 0.0

        self.break_count = 0

        self.previous_time = None

    def update(self, features, timestamp):

        if self.previous_time is None:
            dt = 0.0
        else:
            dt = timestamp - self.previous_time

        self.previous_time = timestamp

        # Ignore large timestamp gaps caused by dropped frames.
        if dt <= 0 or dt > MAX_FRAME_GAP:
            dt = 0.0

        self.history.append((
            timestamp,
            features["torso_angle"],
            features["head_ratio"]
        ))

        # Retain only recent measurements.
        while (
            self.history
            and timestamp - self.history[0][0] > SMOOTHING_SECONDS
        ):
            self.history.popleft()

        # Smooth both features.
        avg_angle = float(np.mean([
            item[1] for item in self.history
        ]))

        avg_head = float(np.mean([
            item[2] for item in self.history
        ]))

        # Hysteresis: different thresholds for entering
        # and leaving the bad posture state.
        bad_condition = (
            avg_angle > BAD_ANGLE
            or avg_head > BAD_HEAD_RATIO
        )

        good_condition = (
            avg_angle < GOOD_ANGLE
            and avg_head < GOOD_HEAD_RATIO
        )

        if self.state == "good":

            if bad_condition:
                self.bad_pending += dt
            else:
                self.bad_pending = 0.0

            if self.bad_pending >= BAD_CONFIRM_SECONDS:

                self.state = "bad"
                self.break_count += 1

                # Reassign the confirmation period.
                self.good_seconds = max(
                    0.0,
                    self.good_seconds - BAD_CONFIRM_SECONDS
                )
                self.bad_seconds += BAD_CONFIRM_SECONDS

                self.bad_pending = 0.0

        else:

            if good_condition:
                self.good_pending += dt
            else:
                self.good_pending = 0.0

            if self.good_pending >= GOOD_CONFIRM_SECONDS:

                self.state = "good"

                self.bad_seconds = max(
                    0.0,
                    self.bad_seconds - GOOD_CONFIRM_SECONDS
                )
                self.good_seconds += GOOD_CONFIRM_SECONDS

                self.good_pending = 0.0

        # Account for the current interval.
        if self.state == "good":
            self.good_seconds += dt
        else:
            self.bad_seconds += dt

        return {
            "state": self.state,
            "smoothed_angle": round(avg_angle, 2),
            "smoothed_head_ratio": round(avg_head, 3)
        }


# --------------------------------------------------
# 4. Session runner
# --------------------------------------------------

def run_monitor(source, duration, output_path, preview=False):

    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError("Could not open video source.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    classifier = PostureClassifier()

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_count = 0
    valid_count = 0
    invalid_count = 0

    start_time = time.monotonic()
    previous_frame_time = None
    last_timestamp = 0.0

    try:

        while True:

            # Stop after the requested duration.
            elapsed = time.monotonic() - start_time

            if elapsed >= duration:
                break

            success, frame = cap.read()

            if not success:
                break

            frame_count += 1

            if frame is None or frame.size == 0:
                invalid_count += 1
                continue

            # Determine video timestamp.
            if isinstance(source, int):
                timestamp = time.monotonic() - start_time
            else:
                timestamp = (
                    cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                )

                if timestamp <= last_timestamp:
                    timestamp = last_timestamp + 1.0 / fps

            last_timestamp = timestamp

            # Ignore corrupt frames.
            if frame.ndim != 3 or frame.shape[2] != 3:
                invalid_count += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = pose.process(rgb)

            if not result.pose_landmarks:
                invalid_count += 1

                if preview:
                    cv2.imshow("Posture Monitor", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                continue

            features = extract_features(
                result.pose_landmarks.landmark
            )

            if features is None:
                invalid_count += 1
                continue

            valid_count += 1

            prediction = classifier.update(
                features,
                timestamp
            )

            if preview:

                mp.solutions.drawing_utils.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp.solutions.pose.POSE_CONNECTIONS
                )

                label = prediction["state"].upper()

                cv2.putText(
                    frame,
                    f"Posture: {label}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0) if label == "GOOD" else (0, 0, 255),
                    2
                )

                cv2.imshow("Posture Monitor", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        pose.close()

        if preview:
            cv2.destroyAllWindows()

    # A usable session must contain enough valid observations.
    if valid_count < 5:
        raise RuntimeError(
            "No usable person observations detected."
        )

    total_time = (
        classifier.good_seconds + classifier.bad_seconds
    )

    if total_time <= 0:
        raise RuntimeError(
            "Insufficient valid observation duration."
        )

    good_percentage = (
        classifier.good_seconds / total_time
    ) * 100

    score = round(good_percentage)

    summary = {
        "session_duration_seconds": round(total_time, 2),
        "frames_processed": frame_count,
        "valid_observations": valid_count,
        "invalid_observations": invalid_count,
        "good_posture_seconds": round(
            classifier.good_seconds, 2
        ),
        "bad_posture_seconds": round(
            classifier.bad_seconds, 2
        ),
        "good_posture_percentage": round(
            good_percentage, 2
        ),
        "posture_breaks": classifier.break_count,
        "posture_score": score,
        "score_method": "Percentage of classified observation time in good posture",
        "thresholds": {
            "good_torso_angle_degrees": GOOD_ANGLE,
            "bad_torso_angle_degrees": BAD_ANGLE,
            "good_head_ratio": GOOD_HEAD_RATIO,
            "bad_head_ratio": BAD_HEAD_RATIO,
            "smoothing_seconds": SMOOTHING_SECONDS,
            "bad_confirmation_seconds": BAD_CONFIRM_SECONDS,
            "good_confirmation_seconds": GOOD_CONFIRM_SECONDS
        }
    }

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    return summary


# --------------------------------------------------
# 5. CLI entry point
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Seated Posture Monitoring System"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Webcam index or video file path"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=60,
        help="Session duration in seconds"
    )

    parser.add_argument(
        "--out",
        default="posture.json",
        help="Output JSON file"
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display optional video preview"
    )

    args = parser.parse_args()

    if args.duration <= 0:
        parser.error("--duration must be greater than zero")

    try:

        summary = run_monitor(
            args.source,
            args.duration,
            args.out,
            args.preview
        )

        print("\nSession completed successfully.")
        print(json.dumps(summary, indent=4))

        return 0

    except Exception as error:

        print(
            f"ERROR: {error}",
            file=sys.stderr
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())
