# Pretrained Models in ML - Futbol

This document describes the pretrained models used in ML-Futbol, what each one does, and how they fit into the overall video analysis pipeline.

## Model Directory Structure

All pretrained models should be placed in the `models/` directory:

```
models/
├── downloaded_pitch.pt # For pitch line/keypoint detection
├── football-player-detection-v9.pt # For detecting players, referees, goalkeepers
├── football-ball-detection-v2.pt # For detecting the ball
```

These models are already trained and ready to use — no training required to run the system.

## 1. football-player-detection-v9.pt
- Type: YOLOv8 Object Detection Model
- Purpose: Detects all human entities on the field:
    - Outfield players
    - Goalkeepers
    - Referees
- Used by: `PlayerDetection` class
- Detection Labels:
    - `player`, `goalkeeper`, `referee` (depending on training data)

This model feeds into the ByteTrack tracker and then into the `TeamClassifier` to assign team IDs.

## 2. football-ball-detection-v2.pt
- Type: YOLOv8 Object Detection Model
- Purpose: Locates the ball in each frame
- Used by: `BallDetection` class
- Annotations: The ball is drawn as a triangle in output videos
- Tracking: Missing frames are interpolated or extrapolated if needed

## 3. downloaded_pitch.pt
- Type: YOLOv8 Object Detection Model
- Purpose: Detects key pitch markers:
    - Corners
    - Penalty spots
    - Center circle
- Used by: `PitchDetection` class
- Output: Field keypoints are used to compute a homography and overlay a pitch layout

## Notes on Model Usage

- All models are loaded via the `Models` class in `ai_futobl_main_classes.py`
- These models are not retrained in this project. If you want to train your own:
    - Use Ultralytics YOLO to train with custom data
    - Replace the .pt files with your own

## Model Versioning

- Model filenames include version numbers (e.g., `v9`, `v2`) to track iterations during development. You may want to:
    - Add notes on performance per version
    - Store older models for ablation studies or comparison
