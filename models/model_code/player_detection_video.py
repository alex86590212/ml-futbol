from auxiliars.team_classifier import TeamClassifier
from tqdm import tqdm
import supervision as sv
import numpy as np
import pandas as pd
import cv2
import os
from config_models import Config, Models


BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    """
    Assigns team IDs to goalkeepers based on their proximity to team player centroids.

    The method assumes each team has a spatial center (centroid), and a goalkeeper is
    classified as belonging to the team whose centroid is closest to them.

    Args:
        players (sv.Detections): Detections of all outfield players, with class_id indicating team.
        goalkeepers (sv.Detections): Detections of goalkeepers (no class/team label yet).

    Returns:
        np.ndarray: Array of assigned team IDs for each goalkeeper (0 or 1).
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) 
    team_0_centroid = players_xy[players.class_id == 0].mean()
    team_1_centroid = players_xy[players.class_id == 1].mean()
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)

class PlayerDetection:
    def __init__(self, config: Config, models: Models, team_classifier: TeamClassifier):
        self.config = config
        self.models = models
        self.team_classifier = team_classifier
        self.player_id = PLAYER_ID
        self.goalkeeper_id = GOALKEEPER_ID
        self.referee_id = REFEREE_ID

        self.video_info = sv.VideoInfo.from_video_path(config.source_video_path)
        self.output_path = config.player_detection_path

        # Annotators
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER
        )

        self.tracker = sv.ByteTrack()
        self.tracker.reset()

        self.stats = {
            "player_detected": 0,
            "player_missing": 0
        }

    def detect_and_save(self):
        """
        Detects, classifies, tracks, and annotates all players, goalkeepers, and referees.

        For each frame:
            - Uses YOLO to detect player-related classes
            - Applies ByteTrack to maintain identity tracking
            - Crops player images and uses the team classifier to assign team IDs
            - Resolves goalkeepers’ teams by proximity to team centroids
            - Normalizes referee labels to a common ID
            - Annotates each detection with ellipses and ID labels
            - Saves annotated frame to video

        Output:
            - Annotated player detection video
            - Printed detection summary
        """
        frame_generator = sv.get_video_frames_generator(self.config.source_video_path)
        video_sink = sv.VideoSink(self.output_path, self.video_info)

        with video_sink:
            for frame_idx, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames, desc="Detecting Players")):
                result = self.models.PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections.with_nms(threshold=0.5, class_agnostic=True)

                # Tracking
                tracked = self.tracker.update_with_detections(detections)

                # Separate classes
                players = tracked[tracked.class_id == self.player_id]
                goalkeepers = tracked[tracked.class_id == self.goalkeeper_id]
                referees = tracked[tracked.class_id == self.referee_id]

                if len(players) > 0:
                    self.stats["player_detected"] += 1
                else:
                    self.stats["player_missing"] += 1

                # Classify players into teams
                player_crops = [sv.crop_image(frame, box) for box in players.xyxy]
                if player_crops:
                    preds = self.team_classifier.predict(player_crops)
                    if len(preds) == len(players):
                        players.class_id = preds
                    else:
                        print(f"⚠️ Mismatch: {len(preds)} labels vs {len(players)} detections")

                # Assign goalkeepers to teams
                goalkeepers.class_id = resolve_goalkeepers_team_id(players, goalkeepers)

                # Normalize referees (class_id = -1)
                referees.class_id -= 1

                # Merge and annotate
                all_detections = sv.Detections.merge([players, goalkeepers, referees])
                labels = [f"#{tid}" for tid in all_detections.tracker_id]
                all_detections.class_id = all_detections.class_id.astype(int)

                annotated = frame.copy()
                annotated = self.ellipse_annotator.annotate(scene=annotated, detections=all_detections)
                annotated = self.label_annotator.annotate(scene=annotated, detections=all_detections, labels=labels)

                video_sink.write_frame(annotated)

        self._log_summary()

    def _log_summary(self):
        """
        Logs the number of frames where players were detected vs. missed.

        Output:
            - Count and percentage of player-detected frames
            - Count and percentage of missed frames
        """
        total = self.video_info.total_frames
        detected = self.stats["player_detected"]
        missing = self.stats["player_missing"]
        print("\n--- Player Detection Summary ---")
        print(f"Total frames: {total}")
        print(f"Players detected in: {detected} frames ({100 * detected / total:.2f}%)")
        print(f"No players detected in: {missing} frames ({100 * missing / total:.2f}%)")