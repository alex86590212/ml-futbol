from ultralytics import YOLO
import gdown
import os
import supervision as sv
import cv2
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoProcessor, SiglipVisionModel
from more_itertools import chunked 
from collections import deque
from team_classifier import TeamClassifier
from pitch_config import SoccerPitchConfiguration
from view_transformer import ViewTransformer
from pitch_draw import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch
from collections import defaultdict
from kalman_filter import KalmanFilter
import matplotlib.pyplot as plt
import pandas as pd

#os.makedirs("example_matches", exist_ok=True)
#url = "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
#url_path = "example_matches/0bfacc_0.mp4"
#url = "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"
#url_path = "/home2/s5549329/ml-futbol/example_matches/121364_0.mp4"
#gdown.download(url, url_path, quiet=False)

#url = "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"
#url_path = "/home2/s5549329/ml-futbol/models/downloaded_pitch.pt"
#gdown.download(url, url_path, quiet=False)
#Input a number of which type of video you want 
NUMBER = 3
TYPE_OF_VIDEO = {0: "None", 1: "Player_Detection_Video", 2: "Pitch_Detection_Video", 3: "All_Detection_Video", 4: "Ball_Detection", 5: "Anotatted_Ball_Detection"}
SOURCE_VIDEO_PATH = "/home2/s5549329/ml-futbol/example_matches/0bfacc_0.mp4"

IMAGES_FOLDER_PATH_PLAYER_DETECTION = "/home2/s5549329/ml-futbol/saved_images/anottated_video.mp4"
IMAGES_FOLDER_PATH_PITCH_DETECTION = "/home2/s5549329/ml-futbol/saved_images/anotatted_pitch_video.mp4"
IMAGES_FOLDER_PATH_PITCH_PLAYER_DETECTION = "/home2/s5549329/ml-futbol/saved_images/anotatted_pitch_player_video.mp4"
IMAGES_FOLDER_BALL_DETECTION_TRACK = "/home2/s5549329/ml-futbol/saved_images/track_ball_video.mp4"
IMAGES_FOLDER_PATH = "/home2/s5549329/ml-futbol/saved_images"
IMAGES_FOLDER_PATH_BALL_DETECTION = "/home2/s5549329/ml-futbol/saved_images/ball_video_detection.mp4"

FIELD_DETECTION_MODEL = YOLO("/home2/s5549329/ml-futbol/models/downloaded_pitch.pt")
PLAYER_DETECTION_MODEL = YOLO("/home2/s5549329/ml-futbol/models/best_player_100.pt")

#!!!!
BALL_DETECTION_MODEL = YOLO("/home2/s5549329/ml-futbol/models/football-ball-detection-v2.pt")
ONLY_PLAYERS_MODEL =YOLO("/home2/s5549329/ml-futbol/models/football-player-detection-v9.pt")

TESTING_MODEL_PITCH = YOLO("/home2/s5549329/runs/detect/pitch-keypoints-yolov12/weights/best.pt")

CONFIG = SoccerPitchConfiguration()

print(os.path.exists(SOURCE_VIDEO_PATH))

# saving the first frame of the video
#frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
#frame = next(frame_generator)

#frame_path = os.path.join(IMAGES_FOLDER_PATH, f"frame_{0:03d}.jpg")
#cv2.imwrite(frame_path, frame)

#anntotations for the players
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
STRIDE = 30

#stride used just for dividing the teams
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=STRIDE)


def interpolate_and_extrapolate_ball_history(ball_history, total_frames, max_interp_gap=10, max_extrap_gap=10, decay=0.85):
    """
    Interpolate and extrapolate ball positions frame-by-frame.

    Args:
        ball_history: List of (frame_idx, x, y)
        total_frames: Total number of frames in the video
        max_interp_gap: Max gap size allowed for interpolation
        max_extrap_gap: Max number of frames to extrapolate
        decay: Velocity decay factor for extrapolation

    Returns:
        Dictionary: {frame_idx: (x, y)}
    """
    ball_records = {frame: (x, y) for frame, x, y in ball_history}
    sorted_frames = sorted(ball_records)
    interpolated = {}

    for frame_idx in range(total_frames):
        if frame_idx in ball_records:
            interpolated[frame_idx] = ball_records[frame_idx]
        else:
            # Find previous and next known frames
            prev = next_ = None
            for f in reversed(sorted_frames):
                if f < frame_idx:
                    prev = f
                    break
            for f in sorted_frames:
                if f > frame_idx:
                    next_ = f
                    break

            if prev is not None and next_ is not None:
                gap = next_ - prev
                if gap <= max_interp_gap:
                    # Interpolate
                    x0, y0 = ball_records[prev]
                    x1, y1 = ball_records[next_]
                    t = (frame_idx - prev) / gap
                    x = x0 + (x1 - x0) * t
                    y = y0 + (y1 - y0) * t
                    interpolated[frame_idx] = (x, y)
            elif prev is not None and next_ is None:
                # Extrapolate forward from last known
                # Estimate velocity from last 2 points
                prev2 = None
                for f in reversed(sorted_frames):
                    if f < prev:
                        prev2 = f
                        break

                if prev2 is not None and (frame_idx - prev) <= max_extrap_gap:
                    x0, y0 = ball_records[prev2]
                    x1, y1 = ball_records[prev]
                    dt = prev - prev2
                    vx = (x1 - x0) / dt
                    vy = (y1 - y0) / dt

                    steps = frame_idx - prev
                    vx *= decay ** steps
                    vy *= decay ** steps

                    x = x1 + vx * steps
                    y = y1 + vy * steps
                    interpolated[frame_idx] = (x, y)

    return interpolated


def interpolate_players_tracker_history(tracker_history, total_frames):
    """Interpolate missing player positions per tracker."""
    interpolated = defaultdict(list)

    for tracker_id, records in tracker_history.items():
        records = sorted(records, key=lambda r: r[0])  # sort by frame
        known = {frame: (x, y, team_id) for frame, x, y, team_id in records}

        for frame_idx in range(total_frames):
            if frame_idx in known:
                interpolated[tracker_id].append((frame_idx, *known[frame_idx]))
            else:
                # Find previous and next known frames
                prev = next = None
                for f in sorted(known):
                    if f < frame_idx:
                        prev = f
                    elif f > frame_idx and next is None:
                        next = f
                if prev is not None and next is not None:
                    x0, y0, team_id = known[prev]
                    x1, y1, _ = known[next]
                    t_ratio = (frame_idx - prev) / (next - prev)
                    x = x0 + (x1 - x0) * t_ratio
                    y = y0 + (y1 - y0) * t_ratio
                    interpolated[tracker_id].append((frame_idx, x, y, team_id))
                # else: no extrapolation
    return interpolated

crops = []
for frame in tqdm(frame_generator, desc="collecting crops"):
    result  = PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)
    detections = detections[detections.class_id == PLAYER_ID]
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    crops += players_crops

team_classifier = TeamClassifier(device="cuda")
team_classifier.fit(crops)
#IDK why is not saving
#crops_image = sv.plot_images_grid(crops[:100], grid_size=(10, 10))
#crops_path = os.path.join(IMAGES_FOLDER_PATH, f"crops_{100:03d}.jpg")
#cv2.imwrite(crops_path, crops_image)

#Resolve assigning goalkeepers to the right team
def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
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

if TYPE_OF_VIDEO[NUMBER] == "Anotatted_Ball_Detection":
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=25,
        height=21,
        outline_thickness=1
    )

    ball_detection_count = 0
    ball_missing_count = 0
    ball_confidence_log = []
    ball_missing_frames = []

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    videosnk = sv.VideoSink(IMAGES_FOLDER_PATH_BALL_DETECTION, video_info)
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    with videosnk:
        for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc="video processing BALL ONLY")):
            result  = BALL_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)

            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            

            if ball_detections is not None and len(ball_detections) > 0:
                ball_detection_count += 1
                for conf in ball_detections.confidence:
                    ball_confidence_log.append(conf)
                    print(f"✅ Frame {frame_idx}: Ball detected with confidence = {conf:.2f}")
            else:
                ball_missing_count += 1
                ball_missing_frames.append(frame_idx)
                print(f"❌ Frame {frame_idx}: Ball not detected")

            annotated_frame = frame.copy()
            annotated_frame = triangle_annotator.annotate(
                scene=annotated_frame,
                detections=ball_detections)
            
            print(f"the shape of the frame: {annotated_frame.shape}")
            videosnk.write_frame(annotated_frame)

    print("\n--- Ball Detection Statistics ---")
    print(f"Total frames: {video_info.total_frames}")
    print(f"Ball detected in frames: {ball_detection_count}")
    print(f"Ball missing in frames: {ball_missing_count}")
    print(f"Detection rate: {100 * ball_detection_count / video_info.total_frames:.2f}%")
    print(f"Missing rate: {100 * ball_missing_count / video_info.total_frames:.2f}%")
    print(f"The frames in which the ball is not being detected: {ball_missing_frames}")

if TYPE_OF_VIDEO[NUMBER] == "Pitch_Detection_Video": 
 
     edge_annotator = sv.EdgeAnnotator(
         color=sv.Color.from_hex('#00BFFF'),
         thickness=2, edges=CONFIG.edges)
     vertex_annotator = sv.VertexAnnotator(
         color=sv.Color.from_hex('#FF1493'),
         radius=8)
     vertex_annotator_2 = sv.VertexAnnotator(
         color=sv.Color.from_hex('#00BFFF'),
         radius=8)
 
 
     video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
     videosnk = sv.VideoSink(IMAGES_FOLDER_PATH_PITCH_DETECTION, video_info)
     frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
 
     with videosnk:
         for frame in tqdm(frame_generator, total=video_info.total_frames, desc="video processing"):
             result = FIELD_DETECTION_MODEL.predict(frame, conf=0.3)[0]
             key_points = sv.KeyPoints.from_ultralytics(result)
             print(key_points.confidence)
 
             filter = key_points.confidence[0] > 0.5
             frame_reference_points = key_points.xy[0][filter]
             frame_reference_key_points = sv.KeyPoints(
                 xy=frame_reference_points[np.newaxis, ...])
 
             pitch_reference_points = np.array(CONFIG.vertices)[filter]
 
             transformer = ViewTransformer(
                 source=pitch_reference_points,
                 target=frame_reference_points
             )
 
             pitch_all_points = np.array(CONFIG.vertices)
             frame_all_points = transformer.transform_points(points=pitch_all_points)
 
             frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])
 
             annotated_frame = frame.copy()
             annotated_frame = edge_annotator.annotate(
                 scene=annotated_frame,
                 key_points=frame_all_key_points)
             annotated_frame = vertex_annotator_2.annotate(
                 scene=annotated_frame,
                 key_points=frame_all_key_points)
             annotated_frame = vertex_annotator.annotate(
                 scene=annotated_frame,
                 key_points=frame_reference_key_points)
             print(annotated_frame.shape)
             videosnk.write_frame(annotated_frame)
 

# The main part where the player, ball, referee, goalkeeper detection is being produced
if TYPE_OF_VIDEO[NUMBER] == "Player_Detection_Video":
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=25,
        height=21,
        outline_thickness=1
    )
    ball_detection_count = 0
    ball_missing_count = 0
    ball_confidence_log = []
    ball_missing_frames = []

    player_detection_count = 0
    player_missing_count = 0

    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    videosnk = sv.VideoSink(IMAGES_FOLDER_PATH_PLAYER_DETECTION, video_info)
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    with videosnk:
        for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc="video processing PLAYER DETECTION")):
            result  = PLAYER_DETECTION_MODEL.predict(frame, conf=0.2)[0]
            detections = sv.Detections.from_ultralytics(result)

            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            if ball_detections is not None and len(ball_detections) > 0:
                ball_detection_count += 1
                for conf in ball_detections.confidence:
                    ball_confidence_log.append(conf)
                    print(f"✅ Frame {frame_idx}: Ball detected with confidence = {conf:.2f}")
            else:
                ball_missing_count += 1
                ball_missing_frames.append(frame_idx)
                print(f"❌ Frame {frame_idx}: Ball not detected")


            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(detections=all_detections)

            goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
            players_detections = all_detections[all_detections.class_id == PLAYER_ID]
            referees_detections = all_detections[all_detections.class_id == REFEREE_ID]
            

            if players_detections is not None and len(players_detections) > 0:
                player_detection_count += 1
                print(f"👥 Frame {frame_idx}: Players detected ({len(players_detections)})")
            else:
                player_missing_count += 1
                print(f"🚫 Frame {frame_idx}: No players detected")

            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            players_detections.class_id = team_classifier.predict(players_crops)

            goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)

            referees_detections.class_id -= 1

            all_detections = sv.Detections.merge([
                players_detections, goalkeepers_detections, referees_detections])

            labels = [
                f"#{tracker_id}"
                for tracker_id
                in all_detections.tracker_id
            ]

            all_detections.class_id = all_detections.class_id.astype(int)

            low_conf_detections = ball_detections[ball_detections.confidence < 0.3]
            high_conf_detections = ball_detections[ball_detections.confidence >= 0.3]

            annotated_frame = frame.copy()
            annotated_frame = ellipse_annotator.annotate(
                scene=annotated_frame,
                detections=all_detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=all_detections,
                labels=labels)
            annotated_frame = sv.TriangleAnnotator(
                color=sv.Color.from_hex('#FF0000'),  # Red = low confidence
                base=25,
                height=21,
                outline_thickness=1
            ).annotate(
                scene=annotated_frame,
                detections=low_conf_detections
            )

            annotated_frame = sv.TriangleAnnotator(
                color=sv.Color.from_hex('#00FF00'),  # Green = high confidence
                base=25,
                height=21,
                outline_thickness=1
            ).annotate(
                scene=annotated_frame,
                detections=high_conf_detections
            )
            print(f"the shape of the frame:{annotated_frame.shape}")
            videosnk.write_frame(annotated_frame)

    print("\n--- Ball Detection Statistics ---")
    print(f"Total frames: {video_info.total_frames}")
    print(f"Ball detected in frames: {ball_detection_count}")
    print(f"Ball missing in frames: {ball_missing_count}")
    print(f"Detection rate: {100 * ball_detection_count / video_info.total_frames:.2f}%")
    print(f"Missing rate: {100 * ball_missing_count / video_info.total_frames:.2f}%")
    print("\n--- Player Detection Statistics ---")
    print(f"Total frames: {video_info.total_frames}")
    print(f"Players detected in frames: {player_detection_count}")
    print(f"Players missing in frames: {player_missing_count}")
    print(f"Detection rate: {100 * player_detection_count / video_info.total_frames:.2f}%")
    print(f"Missing rate: {100 * player_missing_count / video_info.total_frames:.2f}%")
    print(f"The frames in which the ball is not being detected: {ball_missing_frames}")

    ball_confidences = np.array(ball_confidence_log)

    # Filter by confidence ranges
    low_conf = (ball_confidences >= 0.2) & (ball_confidences < 0.3)
    high_conf = ball_confidences >= 0.3

    low_conf_count = np.sum(low_conf)
    high_conf_count = np.sum(high_conf)
    total_detections = len(ball_confidences)

    # Print stats
    print(f"\n--- Ball Confidence Analysis ---")
    print(f"Total ball detections: {total_detections}")
    print(f"Confidence 0.2–0.3: {low_conf_count} ({100 * low_conf_count / total_detections:.2f}%)")
    print(f"Confidence ≥ 0.3  : {high_conf_count} ({100 * high_conf_count / total_detections:.2f}%)")

    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(ball_confidences, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(0.2, color='orange', linestyle='--', label='Lower Threshold = 0.2')
    plt.axvline(0.3, color='red', linestyle='--', label='Current Filter = 0.3')

    # Add text annotations
    plt.text(0.21, plt.ylim()[1]*0.8, f"0.2–0.3: {low_conf_count}", color='orange', fontsize=10)
    plt.text(0.31, plt.ylim()[1]*0.7, f">=0.3: {high_conf_count}", color='red', fontsize=10)

    plt.title('Ball Detection Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Number of Detections')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


#combining both player and pitch detection
if TYPE_OF_VIDEO[NUMBER] == "All_Detection_Video":

    ball_visibility = []

    player_visibility = defaultdict(list) 

    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(f"Video width: {video_info.width}, height: {video_info.height}, fps: {video_info.fps}")
    dt = 1.0 / video_info.fps
    F = np.array([[1, 0, dt, 0],
                   [0, 1, 0, dt],
                   [0, 0, 1,  0],
                   [0, 0, 0,  1]])
    B = np.zeros((4, 2))
    H = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0]])

    Q =  np.diag([0.01, 0.01, 0.5, 0.5])     # very little position noise, some velocity uncertainty  
    R = np.diag([2.0, 2.0])                 # measurement noise: moderate trust in detector  
    P0 = np.eye(4) * 50.0
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    tracker_history = defaultdict(list)
    ball_history = []
    ball_kf = None

    kalman_filters = {}


    for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc="Processing video")):
        if frame is None:
            print(f"Frame {frame_idx} is empty.")
            continue
        print(f"Processing frame {frame_idx}...")

        result = ONLY_PLAYERS_MODEL.predict(frame, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(result)

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
        
        #ball frames appearence
        ball_result = BALL_DETECTION_MODEL.predict(frame, conf=0.3)[0]
        ball_detections = sv.Detections.from_ultralytics(ball_result)
        ball_detections = ball_detections[ball_detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        ball_detected = ball_detections is not None and len(ball_detections) > 0
        if ball_detected:
            ball_visibility.append(True)
            print(f"✅ Frame {frame_idx}: Ball detected ({len(ball_detections)})")
        else:
            ball_visibility.append(False)
            print(f"❌ Frame {frame_idx}: Ball not detected")

        all_detections = detections
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections = tracker.update_with_detections(detections=all_detections)

        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

        #players frames appearence
        detected_ids = set(players_detections.tracker_id if players_detections is not None else [])
        all_known_ids = set(player_visibility.keys()).union(detected_ids)
        for tid in all_known_ids:
            player_visibility[tid].append(tid in detected_ids)

        player_crops = [sv.crop_image(frame, bb) for bb in players_detections.xyxy]
        if len(players_detections) > 0 and player_crops:
            try:
                preds = team_classifier.predict(player_crops)
                if len(preds) == len(players_detections):
                    players_detections.class_id = preds
                else:
                    print(f"Team classifier returned {len(preds)} labels for {len(players_detections)} players, skipping update.")
            except Exception as e:
                print("Team classification error:", e)
        referees_detections.class_id -= 1

        merged_detections = sv.Detections.merge([
            players_detections, goalkeepers_detections, referees_detections
        ])
        labels = [f"#{tid}" for tid in merged_detections.tracker_id]
        merged_detections.class_id = merged_detections.class_id.astype(int)
        
        field_result = FIELD_DETECTION_MODEL.predict(frame, conf=0.3)[0]
        key_points = sv.KeyPoints.from_ultralytics(field_result)

        filter_mask = key_points.confidence[0] > 0.5
        frame_reference_points = key_points.xy[0][filter_mask]
        pitch_reference_points = np.array(CONFIG.vertices)[filter_mask]

        transformer = ViewTransformer(
            source=frame_reference_points,
            target=pitch_reference_points
        )

        frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        if frame_ball_xy is not None and len(frame_ball_xy) > 0:
            raw_ball = transformer.transform_points(points=frame_ball_xy)[0]
        else:
            raw_ball = None

        players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_players_xy = transformer.transform_points(points=players_xy)

        for i, tracker_id in enumerate(players_detections.tracker_id):
            team_id = players_detections.class_id[i]
            player_pos = pitch_players_xy[i]

            # Kalman Filter update
            z = np.array(player_pos).flatten()
            if tracker_id not in kalman_filters:
                kalman_filters[tracker_id] = KalmanFilter(
                    F=F, B=B, H=H, Q=Q, R=R,
                    x0=np.array([z[0], z[1], 0., 0.]),
                    P0=P0
                )
            kf = kalman_filters[tracker_id]
            kf.predict(u=np.zeros(2))
            if z.shape == (2,):
                x = kf.update(z)
            else:
                x = kf.x
            tracker_history[tracker_id].append((frame_idx, x[0], x[1], team_id))
        
        referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_referees_xy = transformer.transform_points(points=referees_xy)

        goalkeepers_xy = goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        pitch_goalkeepers_xy = transformer.transform_points(points=goalkeepers_xy)

        ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        if frame_ball_xy is not None and len(frame_ball_xy) > 0:
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)[0]
            if ball_detected:
                z = transformer.transform_points(points=frame_ball_xy)[0]
                if ball_kf is None:
                    ball_kf = KalmanFilter(F=F, B=B, H=H, Q=Q, R=R, x0=np.array([z[0], z[1], 0., 0.]), P0=P0)
                ball_kf.predict(u=np.zeros(2))
                x = ball_kf.update(z)
                ball_position = (x[0], x[1])
            elif ball_kf is not None:
                ball_kf.predict(u=np.zeros(2))
                x = ball_kf.x
                ball_position = (x[0], x[1])
            else:
                ball_position = None

            if ball_position is not None:
                ball_history.append((frame_idx, ball_position[0], ball_position[1]))



    interpolated_ball_history = interpolate_and_extrapolate_ball_history(
        ball_history,
        video_info.total_frames,
        max_interp_gap=10,
        max_extrap_gap=10,
        decay=0.85
    )
    interpolated_tracker_history = interpolate_players_tracker_history(tracker_history, video_info.total_frames)

    max_len = video_info.total_frames
    for tid, visibility_list in player_visibility.items():
        if len(visibility_list) < max_len:
            player_visibility[tid] += [False] * (max_len - len(visibility_list))

    frame_player_map = defaultdict(dict)
    for tid, records in interpolated_tracker_history.items():
        for frame_idx, x, y, team_id in records:
            frame_player_map[frame_idx][tid] = (x, y, team_id)

    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    interpolated_videosnk = sv.VideoSink(IMAGES_FOLDER_PATH_PITCH_PLAYER_DETECTION, video_info, codec="mp4v")

    with interpolated_videosnk:
        for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc="Rendering combined detected + interpolated frames")):
            annotated_frame = draw_pitch(CONFIG)

            # Players: use real if available, otherwise interpolated
            team0_points, team1_points = [], []
            for tid, (x, y, team_id) in frame_player_map[frame_idx].items():
                if player_visibility.get(tid, [False] * video_info.total_frames)[frame_idx]:
                    real_frames = {rec[0]: rec for rec in tracker_history.get(tid, [])}
                    if frame_idx in real_frames and None not in real_frames[frame_idx][1:3]:
                        x, y, team_id = real_frames[frame_idx][1:]
                if team_id == 0:
                    team0_points.append((x, y))
                elif team_id == 1:
                    team1_points.append((x, y))

            if team0_points:
                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=np.array(team0_points),
                    face_color=sv.Color.from_hex('00BFFF'),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=annotated_frame
                )
            if team1_points:
                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=np.array(team1_points),
                    face_color=sv.Color.from_hex('FF1493'),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=annotated_frame
                )
            # Ball: use real if visible, otherwise interpolated
            if not ball_visibility[frame_idx] and frame_idx in interpolated_ball_history:
                x, y = interpolated_ball_history[frame_idx]
            elif ball_visibility[frame_idx]:
                real = [rec for rec in ball_history if rec[0] == frame_idx and rec[1] is not None]
                if real:
                    x, y = real[0][1], real[0][2]
                else:
                    x, y = None, None
            else:
                x, y = None, None

            if x is not None and y is not None:
                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=np.array([[x, y]]),
                    face_color=sv.Color.from_hex('FFFFFF'),  
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    pitch=annotated_frame
                )

            annotated_frame = cv2.resize(annotated_frame, (video_info.width, video_info.height))
            interpolated_videosnk.write_frame(annotated_frame)


    OUTPUT_FOLDER = "/home2/s5549329/ml-futbol/frames_player_ball"

    print(player_visibility)
    player_df = pd.DataFrame(player_visibility)
    player_df.index.name = "frame_idx"
    player_df = player_df.fillna(False).astype(bool)

    # Optional: save to file
    player_df.to_csv(os.path.join(OUTPUT_FOLDER, "player_visibility.csv"))
    print("✅ Player visibility log saved to 'player_visibility.csv'")

    print(ball_visibility)
    df_ball_visibility = pd.DataFrame({
    "frame_idx": list(range(len(ball_visibility))),
    "ball_visible": ball_visibility
    })
    df_ball_visibility.to_csv(os.path.join(OUTPUT_FOLDER, "ball_visibility.csv"), index=False)
    print("✅ Ball visibility log saved to 'ball_visibility.csv'")

    player_coords = []
    for tracker_id, records in tracker_history.items():
        for frame_idx, x, y, team_id in records:
            player_coords.append({
                "frame_idx": frame_idx,
                "tracker_id": tracker_id,
                "x": x,
                "y": y,
                "team_id": team_id
            })

    df_player_coords = pd.DataFrame(player_coords)
    # Save to desired location
    df_player_coords.to_csv(os.path.join(OUTPUT_FOLDER, "player_coordinates.csv"), index=False)
    print(f"✅ Player coordinates saved to {os.path.join(OUTPUT_FOLDER, 'player_coordinates.csv')}")


elif TYPE_OF_VIDEO[NUMBER] == "Ball_Detection":
    MAXLEN = 5
    BALL_ID = 0
    path_raw = []
    M = deque(maxlen=MAXLEN)

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(f"Video width: {video_info.width}, height: {video_info.height}, fps: {video_info.fps}")
    videosnk = sv.VideoSink(IMAGES_FOLDER_BALL_DETECTION_TRACK, video_info, codec="mp4v")
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    with videosnk:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)

            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            result = FIELD_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            key_points = sv.KeyPoints.from_ultralytics(result)

            filter = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][filter]
            pitch_reference_points = np.array(CONFIG.vertices)[filter]

            transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )
            M.append(transformer.m)
            transformer.m = np.mean(np.array(M), axis=0)

            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            path_raw.append(pitch_ball_xy)

            path = [
                np.empty((0, 2), dtype=np.float32) if coorinates.shape[0] >= 2 else coorinates
                for coorinates
                in path_raw
            ]

            path = [coorinates.flatten() for coorinates in path]
            annotated_frame = draw_pitch(CONFIG)
            annotated_frame = draw_paths_on_pitch(
                config=CONFIG,
                paths=[path],
                color=sv.Color.WHITE,
                pitch=annotated_frame)
            
            if annotated_frame is None or annotated_frame.size == 0:
                print("Warning: annotated_frame is empty, skipping resizing for this frame.")
                continue 
            annotated_frame = cv2.resize(annotated_frame, (video_info.width, video_info.height))
            print(f"Annotated frame shape: {annotated_frame.shape}")
            videosnk.write_frame(annotated_frame)
    videosnk.close()
                
elif NUMBER == 0:
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        thickness=2, edges=CONFIG.edges)
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex('#FF1493'),
        radius=8)
    vertex_annotator_2 = sv.VertexAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        radius=8)

    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    frame = next(frame_generator)

    all_results  = FIELD_DETECTION_MODEL(frame)
    result = FIELD_DETECTION_MODEL.predict(frame, conf=0.3)[0]
    key_points = sv.KeyPoints.from_ultralytics(result)

    for res in all_results:
        xy = res.keypoints.xy
        xyn = res.keypoints.xyn
        kpts = res.keypoints.data

    print(f"the x, y coordinates: {xy}", end=" ")
    print(f"the x, y normalized coordinates: {xyn}", end=" ")
    print(f" the x, y visibility: {kpts}", end=" ")
    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    frame_reference_key_points = sv.KeyPoints(
        xy=frame_reference_points[np.newaxis, ...])

    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=pitch_reference_points,
        target=frame_reference_points
    )

    pitch_all_points = np.array(CONFIG.vertices)
    frame_all_points = transformer.transform_points(points=pitch_all_points)

    frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

    annotated_frame = frame.copy()
    annotated_frame = edge_annotator.annotate(
        scene=annotated_frame,
        key_points=frame_all_key_points)
    annotated_frame = vertex_annotator_2.annotate(
        scene=annotated_frame,
        key_points=frame_all_key_points)
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=frame_reference_key_points)

    frame_path = os.path.join(IMAGES_FOLDER_PATH, f"frame_{0:03d}.jpg")
    cv2.imwrite(frame_path, annotated_frame)

else:
    print("None")

