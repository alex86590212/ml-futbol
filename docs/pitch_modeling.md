# Pitch Modeling in ML - Futbol

This document explains how ML-Futbol detects pitch keypoints and builds a geometric model of the football field.

## Purpose

### Pitch modeling is used to:
- Align the camera view with a real-world coordinate system
- Overlay tactical graphics such as zones, player areas, and Voronoi maps
- Enable analysis like distance to goal, width of press, etc.

## How It Works

### Step 1: Keypoint Detection

- `PitchDetection` uses a YOLO model (`downloaded_pitch.pt`) to detect field keypoints:
    - Penalty spots
    - Center circle
    - Corner flags
    - Box corners

### Step 2: Homography Calculation
- Detected keypoints are matched to a canonical pitch layout
- A homography matrix is computed using OpenCV
- This matrix transforms image coordinates into pitch coordinates

### Step 3: Pitch Overlay
- The homography is passed to the `ViewTransformer`
- The pitch layout is drawn using points and lines from `pitch_config.py`
- Optional: Voronoi maps and tactical zones can be drawn in `pitch_draw.py`

## Output
The result is a video saved as:
- `saved_images/anotatted_pitch_video.mp4`

This includes:
- Detected pitch keypoints
- Transformed pitch model overlaid on each frame