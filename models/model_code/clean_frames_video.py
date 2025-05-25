from auxiliars.team_classifier import TeamClassifier
from tqdm import tqdm
import supervision as sv
import numpy as np
import pandas as pd
import cv2
import os
from config_models import Config, Models

PLAYER_THRESHOLD = 10
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

class CleanVideo:
    def __init__(self, config: Config, models: Models):
        self.config = config
        self.models = models
        self.player_id = PLAYER_ID
        self.goalkeeper_id = GOALKEEPER_ID
        self.referee_id = REFEREE_ID

        self.video_info = sv.VideoInfo.from_video_path(config.source_video_path)
        self.output_path = config.clean_video_path

        self.tracker = sv.ByteTrack()
        self.tracker.reset()

        self.stats = {
            "player_detected": 0,
            "player_missing": 0,
            "frames_kept": 0,
            "frames_skipped": 0
        }

    def detect_and_save(self):
        frame_generator = sv.get_video_frames_generator(self.config.source_video_path)
        video_sink = sv.VideoSink(self.output_path, self.video_info)

        with video_sink:
            for frame_idx, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames, desc="Detecting Players")):
                result = self.models.PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections.with_nms(threshold=0.5, class_agnostic=True)

                tracked = self.tracker.update_with_detections(detections)

                players = tracked[tracked.class_id == self.player_id]
                goalkeepers = tracked[tracked.class_id == self.goalkeeper_id]

                total_detected = len(players) + len(goalkeepers)
                print(f"Frame {frame_idx}: {total_detected} players detected")

                if len(players) > 0:
                    self.stats["player_detected"] += 1
                else:
                    self.stats["player_missing"] += 1

                if len(players) + len(goalkeepers) < PLAYER_THRESHOLD:
                    self.stats["frames_skipped"] += 1
                    continue
                else:
                    self.stats["frames_kept"] += 1

                # Write unannotated (raw) frame
                video_sink.write_frame(frame)

        self._log_summary()

    def _log_summary(self):
        total = self.video_info.total_frames
        detected = self.stats["player_detected"]
        missing = self.stats["player_missing"]
        kept = self.stats["frames_kept"]
        skipped = self.stats["frames_skipped"]

        print("\n--- Player Detection Summary ---")
        print(f"Total frames: {total}")
        print(f"Players detected in: {detected} frames ({100 * detected / total:.2f}%)")
        print(f"No players detected in: {missing} frames ({100 * missing / total:.2f}%)")
        print(f"Frames kept (>= {PLAYER_THRESHOLD} players): {kept} ({100 * kept / total:.2f}%)")
        print(f"Frames skipped (< {PLAYER_THRESHOLD} players): {skipped} ({100 * skipped / total:.2f}%)")