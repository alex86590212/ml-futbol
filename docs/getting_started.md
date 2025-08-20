# Getting Started with ML - Futbol

Welcome to the **ML-Futbol** framework! This guide will help you install dependencies and set up your environment.

## 1. Installation

### 1.1 Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### 1.2 Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Dataset & Model Setup

### 2.1 Input Videos

Make sure your match videos are placed in the following folder:

```
example_matches/
├── 0bfacc_0.mp4
├── 121364_0.mp4
```

### 2.2 Pretrained Models

Place the pretrained YOLO models in the following directory:

```
models/
├── downloaded_pitch.pt
├── football-player-detection-v9.pt
├── football-ball-detection-v2.pt
```

## 3. First Run (Player Detection)

To run the player detection and save an annotated video:

```python
from ai_futobl_main_classes import Config, Models, PlayerDetection
from team_classifier import TeamClassifier

config = Config(NUMBER=1)  # 1 = Player Detection
models = Models()
classifier = TeamClassifier(device="cuda")
detector = PlayerDetection(config, models, classifier)
detector.detect_and_save()
```

This will generate:  

`saved_images/anottated_player_video.mp4`

## 4. Other Detection Modes

Use the `Config(NUMBER=...)` to change modes:

```text
| Mode Number | Description                   |
|-------------|-------------------------------|
| 1           | Player Detection              |
| 2           | Pitch Detection               |
| 3           | Combined Player + Pitch       |
| 4           | Ball Detection                |
```

## 5️. Output Structure

### 5.1 Annotated Videos

Saved in the `saved_images/` folder:

```
saved_images/
├── anottated_player_video.mp4
├── anotatted_pitch_video.mp4
├── ball_video_detection.mp4
├── kalman.mp4
├── track_ball_video.mp4
```

### 5.2 Intermediate CSV Data

Stored in the `frames_player_ball/` folder:

```
frames_player_ball/
├── ball_visibility.csv
├── player_coordinates.csv
├── player_visibility.csv
```

## 6️. Next Steps

- See [usage.md](usage.md) for more ways to run the system.
- Explore [architecture.md](architecture.md) to understand the internals.
- View output examples in [outputs.md](outputs.md)
