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

TYPE_OF_VIDEO = {0: "None", 1: "Player_Detection_Video", 2: "Pitch_Detection_Video", 3: "All_Detection_Video", 4: "Ball_Detection"}
SOURCE_VIDEO_PATH = "/home4/s5539099/test/ml-futbol/example_matches/121364_0.mp4"

IMAGES_FOLDER_PATH_PLAYER_DETECTION = "/home4/s5539099/test/ml-futbol/saved_images/anottated_video.mp4"
IMAGES_FOLDER_PATH_PITCH_DETECTION = "/home4/s5539099/test/ml-futbol/saved_images/anotatted_pitch_video.mp4"
IMAGES_FOLDER_PATH_PITCH_PLAYER_DETECTION = "/home4/s5539099/test/ml-futbol/saved_images/anotatted_pitch_player_video.mp4"
IMAGES_FOLDER_BALL_DETECTION = "/home4/s5539099/test/ml-futbol/saved_images/ball_video.mp4"
IMAGES_FOLDER_PATH = "/home4/s5539099/test/ml-futbol/saved_images"

FIELD_DETECTION_MODEL = YOLO("/home4/s5539099/test/ml-futbol/models/downloaded_pitch.pt")
PLAYER_DETECTION_MODEL = YOLO("/home4/s5539099/test/ml-futbol/models/best_player_100.pt")

CONFIG = SoccerPitchConfiguration()

print(os.path.exists(SOURCE_VIDEO_PATH))

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
STRIDE = 30

#stride used just for dividing the teams
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=STRIDE)
# ----- Kalman Filter for 2D smoothing -----
class KalmanFilter2D:
    def __init__(self, dt=1.0, process_var=1.0, meas_var=1.0):
        # State: [x, y, vx, vy]
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1,  0],
                           [0, 0, 0,  1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = process_var * np.eye(4)
        self.R = meas_var * np.eye(2)
        self.P = np.eye(4)
        self.x = np.zeros((4,1))
        self.initialized = False

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, meas):
        z = meas.reshape((2,1))
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

    def smooth(self, meas):
        if not self.initialized:
            # Initialize state on first measurement
            self.x[0,0], self.x[1,0] = meas
            self.initialized = True
        else:
            self.predict()
            self.update(meas)
        return self.x[0,0], self.x[1,0]


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

    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    videosnk = sv.VideoSink(IMAGES_FOLDER_PATH_PLAYER_DETECTION, video_info)
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    with videosnk:
        for frame in tqdm(frame_generator, total=video_info.total_frames, desc="video processing PLAYER DETECTION"):
            result  = PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)

            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(detections=all_detections)

            goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
            players_detections = all_detections[all_detections.class_id == PLAYER_ID]
            referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

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

            annotated_frame = frame.copy()
            annotated_frame = ellipse_annotator.annotate(
                scene=annotated_frame,
                detections=all_detections)
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=all_detections,
                labels=labels)
            annotated_frame = triangle_annotator.annotate(
                scene=annotated_frame,
                detections=ball_detections)
            print(f"the shape of the frame:{annotated_frame.shape}")
            videosnk.write_frame(annotated_frame)

#combining both player and pitch detection
if TYPE_OF_VIDEO[NUMBER] == "All_Detection_Video":
    # 1) Initialize tracker, video info & sink
    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(f"Video width: {video_info.width}, height: {video_info.height}, fps: {video_info.fps}")
    videosnk = sv.VideoSink(IMAGES_FOLDER_PATH_PITCH_PLAYER_DETECTION, video_info, codec="mp4v")
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    frame_count = 0

    # 2) Prepare Kalman filters
    kalman_filters = {}                                     # one filter per player tracker_id
    ball_kf = KalmanFilter2D(process_var=0.5, meas_var=2.0) # single filter for the ball

    with videosnk:
        for frame_idx, frame in enumerate(tqdm(frame_generator,
                                              total=video_info.total_frames,
                                              desc="Processing video")):
            if frame is None:
                continue

            # --- DETECTION & TRACKING ---
            res = PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            dets = sv.Detections.from_ultralytics(res)

            # ball
            ball_dets = dets[dets.class_id == BALL_ID]
            ball_dets.xyxy = sv.pad_boxes(ball_dets.xyxy, px=10)

            # players, goalkeepers, referees
            others = dets[dets.class_id != BALL_ID].with_nms(threshold=0.5, class_agnostic=True)
            tracked = tracker.update_with_detections(detections=others)
            ply_dets = tracked[tracked.class_id == PLAYER_ID]
            gk_dets  = tracked[tracked.class_id == GOALKEEPER_ID]
            ref_dets = tracked[tracked.class_id == REFEREE_ID]
            ref_dets.class_id -= 1

            # --- TEAM CLASSIFICATION & GK ASSIGNMENT ---
            if len(ply_dets):
                crops = [sv.crop_image(frame, xy) for xy in ply_dets.xyxy]
                ply_dets.class_id = team_classifier.predict(crops)
            if len(gk_dets):
                gk_dets.class_id = resolve_goalkeepers_team_id(ply_dets, gk_dets)

            # --- PITCH HOMOGRAPHY ---
            fld_res = FIELD_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            kpts    = sv.KeyPoints.from_ultralytics(fld_res)
            mask    = kpts.confidence[0] > 0.5
            src_pts = kpts.xy[0][mask]
            dst_pts = np.array(CONFIG.vertices)[mask]
            transformer = ViewTransformer(source=src_pts, target=dst_pts)

            # --- TRANSFORM TO PITCH COORDINATES ---
            raw_player_pts = ply_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_player_pts = transformer.transform_points(raw_player_pts)

            raw_ball_pts = ball_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_pts = transformer.transform_points(raw_ball_pts) if len(raw_ball_pts) > 0 else []

            # --- KALMAN SMOOTHING ---
            team0_points = []
            team1_points = []
            for tid, meas, team in zip(ply_dets.tracker_id, pitch_player_pts, ply_dets.class_id):
                if tid not in kalman_filters:
                    kalman_filters[tid] = KalmanFilter2D(process_var=0.5, meas_var=2.0)
                x_s, y_s = kalman_filters[tid].smooth(meas)
                if team == 0:
                    team0_points.append([x_s, y_s])
                else:
                    team1_points.append([x_s, y_s])

            ball_point = []
            if len(pitch_ball_pts) > 0:
                bx, by = ball_kf.smooth(pitch_ball_pts[0])
                ball_point = [[bx, by]]

            # --- DRAW ON PITCH & WRITE FRAME ---
            annotated = draw_pitch(CONFIG)
            if team0_points:
                annotated = draw_points_on_pitch(
                    config=CONFIG,
                    xy=np.array(team0_points),
                    face_color=sv.Color.from_hex('00BFFF'),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=annotated
                )
            if team1_points:
                annotated = draw_points_on_pitch(
                    config=CONFIG,
                    xy=np.array(team1_points),
                    face_color=sv.Color.from_hex('FF1493'),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=annotated
                )
            if ball_point:
                annotated = draw_points_on_pitch(
                    config=CONFIG,
                    xy=np.array(ball_point),
                    face_color=sv.Color.from_hex('FFFFFF'),
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    pitch=annotated
                )

            annotated = cv2.resize(annotated, (video_info.width, video_info.height))
            videosnk.write_frame(annotated)
            frame_count += 1
    print(f"Wrote {frame_count} frames to {IMAGES_FOLDER_PATH_PITCH_PLAYER_DETECTION}")
    videosnk.close()



elif TYPE_OF_VIDEO[NUMBER] == "Ball_Detection":
    MAXLEN = 5
    BALL_ID = 0
    path_raw = []
    M = deque(maxlen=MAXLEN)

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(f"Video width: {video_info.width}, height: {video_info.height}, fps: {video_info.fps}")
    videosnk = sv.VideoSink(IMAGES_FOLDER_BALL_DETECTION, video_info, codec="mp4v")
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

    result = FIELD_DETECTION_MODEL.predict(frame, conf=0.3)[0]
    key_points = sv.KeyPoints.from_ultralytics(result)

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
