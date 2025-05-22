# Developer Guide

This document outlines the key public classes and methods provided by ML-Futbol, especially those intended for use in scripts, notebooks, or custom pipelines.

## Core Classes (from `ai_futobl_main_classes.py`)

### `Config`
```
Config(NUMBER: int)
```
- Initializes file paths and output folders
- Selects the type of detection task based on `NUMBER` (1 to 4)

**Attributes:**
- `source_video_path`: path to the input match video  
- `output_folder`: where outputs are stored  

---

### `Models`
```
Models()
```
- Loads all required YOLOv8 models

**Attributes:**
- `player_detector`: model for player detection  
- `ball_detector`: model for ball detection  
- `field_detector`: model for pitch detection  

---

### `PlayerDetection`
```
PlayerDetection(config, models, classifier)
```
- Detects and tracks players
- Applies team classification using the classifier

**Methods:**
- `detect_and_save()`: runs the full detection pipeline and saves the annotated output video

---

### `BallDetection`
```
BallDetection(config, models)
```
- Detects and tracks the football throughout the match

**Methods:**
- `detect_and_save()`: generates an annotated video showing ball detection and movement

---

### `PitchDetection`
```
PitchDetection(config, models)
```
- Detects pitch lines and keypoints using YOLO  
- Computes homography and overlays a 2D pitch model

**Methods:**
- `detect_and_save()`: runs detection and saves an annotated video with the pitch layout

---

## Team Classifier (from `team_classifier.py`)

### `TeamClassifier`
```
TeamClassifier(device="cuda")
```
- Loads a SigLIP vision model  
- Embeds visual player crops  
- Clusters features to assign team IDs

**Methods:**
- `extract_features(player_images: list) -> np.ndarray`: returns feature vectors for input crops  
- `cluster_features(features: np.ndarray) -> list`: clusters features into two teams and returns team labels

---

## View Transformer (from `view_transformer.py`)

### `ViewTransformer`
```
ViewTransformer(homography: np.ndarray)
```
- Handles projection between camera view and pitch view using a homography matrix

**Methods:**
- `transform(point: np.ndarray) -> np.ndarray`: project a point from image to pitch space  
- `inverse_transform(point: np.ndarray) -> np.ndarray`: revert pitch space back to image space
