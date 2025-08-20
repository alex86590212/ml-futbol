# Usage Guide for ML - Futbol

This guide explains how to run the different video analysis modes supported by ML-Futbol using the class-based interface.

## 1. Prerequisites

Ensure you have followed the steps in [getting_started.md](getting_started.md):
- Dependencies are installed.
- Match videos are placed in `example_matches/`
- Pretrained YOLO models are in `models/`

## 2️. Select a Detection Mode

Each run mode corresponds to a specific video analysis task:

| Mode Number | Detection Type           | Class Used         |
|-------------|---------------------------|---------------------|
| 1           | Player Detection          | `PlayerDetection`   |
| 2           | Pitch Detection           | `PitchDetection`    |
| 3           | Combined Player + Pitch   | `PitchDetection` + `PlayerDetection` |
| 4           | Ball Detection            | `BallDetection`     |

Use `Config(NUMBER=...)` to select the mode.

## 3. Player Detection

Tracks all players, assigns team identities, and saves an annotated video.

```python
from ai_futobl_main_classes import Config, Models, PlayerDetection
from team_classifier import TeamClassifier

config = Config(NUMBER=1)
models = Models()
classifier = TeamClassifier(device="cuda")
detector = PlayerDetection(config, models, classifier)
detector.detect_and_save()
```

Output: `saved_images/anottated_player_video.mp4`

## 4. Pitch Detection

Detects pitch keypoints and overlays a wireframe model of the field.

```python
from ai_futobl_main_classes import Config, Models, PitchDetection

config = Config(NUMBER=2)
models = Models()
detector = PitchDetection(config, models)
detector.detect_and_save()
```

Output: `saved_images/anotatted_pitch_video.mp4`

## 5. Combined Player + Pitch Detection

Detects both the players and pitch overlay in a single frame-by-frame render. Run both modules one after the other.

```python
from ai_futobl_main_classes import Config, Models, PlayerDetection, PitchDetection
from team_classifier import TeamClassifier

config = Config(NUMBER=3)
models = Models()
classifier = TeamClassifier(device="cuda")

pitch_detector = PitchDetection(config, models)
pitch_detector.detect_and_save()

player_detector = PlayerDetection(config, models, classifier)
player_detector.detect_and_save()
```

Outputs:

- `saved_images/anotatted_pitch_video.mp4`
- `saved_images/anottated_player_video.mp4`

## 6. Ball Detection

Tracks the ball, annotates it with a triangle, and handles missing frame recovery

```Python
from ai_futobl_main_classes import Config, Models, BallDetection

config = Config(NUMBER=4)
models = Models()
detector = BallDetection(config, models)
detector.detect_and_save()
```

Output: `saved_images/ball_video_detection.mp4`

## 7. See Also:

- See [getting_started.md](getting_started.md) Install and run the first test.
- Explore [architecture.md](architecture.md) How the system is structured.
- View output examples in [outputs.md](outputs.md) How the output videos and CSVs look like.



