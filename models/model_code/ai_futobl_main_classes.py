from ultralytics import YOLO
from auxiliars.team_classifier import TeamClassifier
from tqdm import tqdm
import supervision as sv
from collections import defaultdict
import numpy as np
from auxiliars.pitch_draw import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch
from auxiliars.pitch_config import SoccerPitchConfiguration
from auxiliars.view_transformer import ViewTransformer
from auxiliars.kalman_filter import KalmanFilter
import pandas as pd
import cv2
import os
import hydra
from omegaconf import DictConfig, OmegaConf

from all_detection_video import AllDetectionVideo
from ball_detection_video import BallDetection
from pitch_detection_video import PitchDetection
from player_detection_video import PlayerDetection
from config_models import Config, Models

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

@hydra.main(config_path="/home2/s5549329/scripts/ml-futbol/configs", config_name="kalman", version_base=None)
def main(cfg: DictConfig):
    # Initialize config, models, and pitch configuration
    config = Config(NUMBER=1)  # 3 corresponds to "All_Detection_Video"
    models = Models()
    pitch_config = SoccerPitchConfiguration()
    #{0: "None", 1: "Player_Detection_Video", 2: "Pitch_Detection_Video", 3: "All_Detection_Video", 4: "Anotatted_Ball_Detection"}
    if config.type_of_video in ["Player_Detection_Video", "All_Detection_Video"]:
        frame_generator = sv.get_video_frames_generator(config.source_video_path)
        crops = []
        for frame in tqdm(frame_generator, desc="collecting crops"):
            result  = models.PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            detections = detections[detections.class_id == PLAYER_ID]
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
            crops += players_crops

        team_classifier = TeamClassifier(device="cuda")
        team_classifier.fit(crops)

    # Step 2: Run the All Detection pipeline
    if config.type_of_video == "All_Detection_Video":
        B = np.array(cfg.kalman.B)
        H = np.array(cfg.kalman.H)
        Q = np.diag(cfg.kalman.Q)
        R = np.diag(cfg.kalman.R)
        P0 = np.eye(4) * cfg.kalman.P0

        kalman_params = {
        "B": B,
        "H": H,
        "Q": Q,
        "R": R,
        "P0": P0
        }

        pipeline = AllDetectionVideo(config=config, models=models, team_classifier=team_classifier, CONFIG=pitch_config, kalman_params=kalman_params)

        # Step 3: Process video and interpolate missing data
        pipeline.detect_and_interpolate()

        # Step 4: Render the annotated output video
        pipeline.render_interpolated_video()

        # Step 5: Export logs
        output_folder = "ml-futbol/frames_player_ball"
        pipeline.export_tracking_tables(output_folder=output_folder)


    elif config.type_of_video == "Player_Detection_Video":
        detector = PlayerDetection(config=config, models=models, team_classifier=team_classifier)
        detector.detect_and_save()
        detector._log_summary()


    elif config.type_of_video == "Pitch_Detection_Video":
        detector = PitchDetection(config=config, models=models, CONFIG=pitch_config)
        detector.detect_and_save()

    elif config.type_of_video == "Anotatted_Ball_Detection":
        detector = BallDetection(config=config, models=models)
        detector.detect_and_save()
        detector._log_summary()
    
    else:
        print(f"⚠️ Unknown or unsupported video type: {config.type_of_video}")


if __name__ == "__main__":
    main()