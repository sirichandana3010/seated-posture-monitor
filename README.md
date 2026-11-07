
# Seated Posture Monitor

A lightweight CPU-based seated posture monitoring system using Python, OpenCV and MediaPipe Pose.

## Features

- Supports webcam and video file input.
- Detects upper-body landmarks using MediaPipe.
- Extracts torso angle and normalized head displacement.
- Uses temporal smoothing and hysteresis to reduce classification flicker.
- Requires sustained posture changes before registering transitions.
- Generates a JSON session summary.
- Supports optional preview mode.
- Returns non-zero exit codes for unusable observations.

## Setup

Python 3.10+ is required.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Webcam:

```bash
python posture_monitor.py --source 0 --duration 60 --out posture.json
```

Video:

```bash
python posture_monitor.py --source sitting.mp4 --duration 60 --out posture.json
```

Optional preview:

```bash
python posture_monitor.py --source 0 --duration 60 --preview
```

## Technical Decisions

- MediaPipe Pose was selected for pretrained landmark detection and CPU-based inference.
- Torso angle measures upper-body inclination relative to the image's vertical axis.
- Head displacement is normalized by torso length to reduce sensitivity to camera distance.
- A 0.5-second moving window smooths noisy measurements.
- Hysteresis uses separate entry and recovery thresholds.
- A bad posture must persist for 1.5 seconds; recovery requires 1 second.
- The initial thresholds are engineering assumptions and should be calibrated using representative recordings.
- The posture score represents the percentage of classified observation time spent in good posture.

## Limitations

- Assumes a side or slightly oblique camera position.
- Uses 2D landmarks and does not estimate true 3D spinal alignment.
- Thresholds are not medically validated.
- Missing-person intervals are excluded from classified observation time.
- Does not support multiple people or detailed ergonomic assessment.
