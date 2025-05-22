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

class BallDetection:
    def __init__(self, config: Config, models: Models):
        self.config = config
        self.models = models
        self.ball_id = BALL_ID
        self.video_info = sv.VideoInfo.from_video_path(config.source_video_path)
        self.output_path = config.ball_detection_path
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        )

        self.detection_stats = {
            "detected": 0,
            "missing": 0,
            "confidence_log": [],
            "missing_frames": []
        }

    def detect_and_save(self):
        """
        Detects the ball in each video frame and writes an annotated output video.

        For every frame, runs ball detection using the YOLO model and annotates the
        detected ball (if found) using a triangle marker. It also logs detection stats
        including missing frames and confidence levels, and outputs a summary.

        Output:
            - Annotated video saved to self.output_path
            - Printed summary of detection rate and confidence distribution
        """

        frame_generator = sv.get_video_frames_generator(self.config.source_video_path)
        video_sink = sv.VideoSink(self.output_path, self.video_info)

        with video_sink:
            for frame_idx, frame in enumerate(
                tqdm(frame_generator, total=self.video_info.total_frames, desc="Detecting Ball")
            ):
                result = self.models.BALL_DETECTION_MODEL.predict(frame, conf=0.3)[0]
                detections = sv.Detections.from_ultralytics(result)
                ball_detections = detections[detections.class_id == self.ball_id]
                ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

                if ball_detections and len(ball_detections) > 0:
                    self.detection_stats["detected"] += 1
                    self.detection_stats["confidence_log"].extend(ball_detections.confidence.tolist())
                    print(f"✅ Frame {frame_idx}: Ball detected")
                else:
                    self.detection_stats["missing"] += 1
                    self.detection_stats["missing_frames"].append(frame_idx)
                    print(f"❌ Frame {frame_idx}: Ball not detected")

                annotated = self.triangle_annotator.annotate(scene=frame.copy(), detections=ball_detections)
                video_sink.write_frame(annotated)

        self._log_summary()

    def _log_summary(self):
        """
        Prints a summary of the ball detection process.

        Displays:
            - Total number of frames processed
            - Frames where the ball was detected or missed
            - Confidence distribution of the detections
            - List of missing frame indices
        """
        total = self.video_info.total_frames
        detected = self.detection_stats["detected"]
        missing = self.detection_stats["missing"]
        print("\n--- Ball Detection Summary ---")
        print(f"Total frames: {total}")
        print(f"Detected: {detected} ({100 * detected / total:.2f}%)")
        print(f"Missing: {missing} ({100 * missing / total:.2f}%)")
        print(f"Missing frames: {self.detection_stats['missing_frames']}")

        if self.detection_stats["confidence_log"]:
            confs = np.array(self.detection_stats["confidence_log"])
            low = confs[(confs >= 0.2) & (confs < 0.3)]
            high = confs[confs >= 0.3]
            print(f"\nConfidence 0.2–0.3: {len(low)} ({100 * len(low) / len(confs):.2f}%)")
            print(f"Confidence ≥ 0.3: {len(high)} ({100 * len(high) / len(confs):.2f}%)")