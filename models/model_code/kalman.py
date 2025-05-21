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

def smooth_trajectory(raw_traj, F, B, H, Q, R, P0):
    """
    raw_traj: list of (frame_idx, x, y, team_flag)
    Returns: list of (frame_idx, x_smooth, y_smooth, team_flag)
    """
    if not raw_traj:
        return []

    raw_traj = sorted(raw_traj, key=lambda x: x[0])
    # Initialize x & y filters
    x0 = np.array([[raw_traj[0][1]], [0.0]])
    y0 = np.array([[raw_traj[0][2]], [0.0]])
    kf_x = KalmanFilter(F, B, H, Q, R, x0.copy(), P0.copy())
    kf_y = KalmanFilter(F, B, H, Q, R, y0.copy(), P0.copy())

    smoothed = []
    prev_frame = raw_traj[0][0]
    for frame_idx, raw_x, raw_y, team in raw_traj:
        gap = frame_idx - prev_frame
        # Predict through any skipped frames
        for _ in range(gap):
            kf_x.predict(u=np.zeros((1,1)))
            kf_y.predict(u=np.zeros((1,1)))
        # Measurement update
        kf_x.update(np.array([[raw_x]]))
        kf_y.update(np.array([[raw_y]]))

        xs = float(kf_x.x[0])
        ys = float(kf_y.x[0])
        smoothed.append((frame_idx, xs, ys, team))
        prev_frame = frame_idx

    return smoothed

def interpolate_tracker_history(tracker_history, fractions=[0.25, 0.5, 0.75]):
    """
    For each tracked player, interpolate extra points between successive frames.
    
    Parameters:
        tracker_history (dict): Dictionary where key is tracker_id and value is a list of tuples:
                                (frame_index, x, y, team_id).
        fractions (list): List of fractions in (0,1) at which to interpolate between two frames.
    
    Returns:
        dict: A new dictionary with the same structure but with extra interpolated points.
    """
    new_tracker_history = {}
    for tracker_id, traj in tracker_history.items():
        traj = sorted(traj, key=lambda x: x[0])
        new_traj = []
        for i in range(len(traj) - 1):
            t0, x0, y0, team_id = traj[i]
            t1, x1, y1, _ = traj[i + 1]

            # Add the starting point
            new_traj.append((t0, x0, y0, team_id))
            
            dt = t1 - t0
            for frac in fractions:
                new_t = t0 + frac * dt
                new_x = x0 + frac * (x1 - x0)
                new_y = y0 + frac * (y1 - y0)
                new_traj.append((new_t, new_x, new_y, team_id))
        
        # Append the final point of the trajectory
        new_traj.append(traj[-1])
        new_tracker_history[tracker_id] = new_traj
    return new_tracker_history

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
    Q = np.eye(4) * 0.01
    R = np.eye(2) * 10.0
    P0 = np.eye(4) * 500.0

    videosnk = sv.VideoSink(IMAGES_FOLDER_PATH_PITCH_PLAYER_DETECTION, video_info, codec="mp4v")
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    tracker_history = defaultdict(list)
    ball_history = []
    kalman_filters = {}
    with videosnk:
        for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames, desc="Processing video")):
            if frame is None:
                print(f"Frame {frame_idx} is empty.")
                continue
            print(f"Processing frame {frame_idx}...")

            result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)

            ball_detections = detections[detections.class_id == BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            all_detections = detections[detections.class_id != BALL_ID]
            all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
            all_detections = tracker.update_with_detections(detections=all_detections)

            goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
            players_detections = all_detections[all_detections.class_id == PLAYER_ID]
            referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

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
                pitch_ball_raw = transformer.transform_points(points=frame_ball_xy)
                if pitch_ball_raw.shape[0] > 0:
                    pitch_ball_xy = pitch_ball_raw[0]
                else:
                    pitch_ball_xy = None
            else:
                pitch_ball_xy = None

            players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_players_xy = transformer.transform_points(points=players_xy)
            
            referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_referees_xy = transformer.transform_points(points=referees_xy)

            goalkeepers_xy = goalkeepers_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_goalkeepers_xy = transformer.transform_points(points=goalkeepers_xy)

            # Update tracker history with the (frame_idx, x, y, team_id) for each player.
            for i, tracker_id in enumerate(players_detections.tracker_id):
                team_id = players_detections.class_id[i]
                player_pos = pitch_players_xy[i]
                tracker_history[tracker_id].append((frame_idx, player_pos[0], player_pos[1], team_id))

            ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            if ball_xy is not None and len(ball_xy) > 0:
                pitch_ball_xy = transformer.transform_points(points=ball_xy)[0]
                ball_history.append((frame_idx, pitch_ball_xy[0], pitch_ball_xy[1], -1))

            for i, tid in enumerate(players_detections.tracker_id):
                z = np.array(pitch_players_xy[i]).flatten()
                if z.shape == (2,):
                    if tid not in kalman_filters:
                        kalman_filters[tid] = KalmanFilter(F, B, H, Q, R, np.array([z[0], z[1], 0.0, 0.0]), P0)
                    kf = kalman_filters[tid]
                    kf.predict(u=np.zeros(2))
                    x = kf.update(z)
                    pitch_players_xy[i] = x[:2]
                    tracker_history[tid].append((frame_idx, x[0], x[1], players_detections.class_id[i]))

            # Referees
            for i, tid in enumerate(referees_detections.tracker_id):
                z = np.array(pitch_referees_xy[i]).flatten()
                if z.shape == (2,):
                    if tid not in kalman_filters:
                        kalman_filters[tid] = KalmanFilter(F, B, H, Q, R, np.array([z[0], z[1], 0.0, 0.0]), P0)
                    kf = kalman_filters[tid]
                    kf.predict(u=np.zeros(2))
                    x = kf.update(z)
                    pitch_referees_xy[i] = x[:2]
                    tracker_history[tid].append((frame_idx, x[0], x[1], referees_detections.class_id[i]))

             # Goalkeepers
            for i, tid in enumerate(goalkeepers_detections.tracker_id):
                z = np.array(pitch_goalkeepers_xy[i]).flatten()
                if z.shape == (2,):
                    if tid not in kalman_filters:
                        kalman_filters[tid] = KalmanFilter(
                            F, B, H, Q, R,
                            np.array([z[0], z[1], 0.0, 0.0]),
                            P0
                        )
                    kf = kalman_filters[tid]
                    kf.predict(u=np.zeros(2))
                    x = kf.update(z)
                    pitch_goalkeepers_xy[i] = x[:2]
                    tracker_history[tid].append((frame_idx, x[0], x[1], goalkeepers_detections.class_id[i]))

            
            # Ball
            if pitch_ball_xy is not None:
                z = np.array(pitch_ball_xy).flatten()
                if z.shape == (2,):
                    if 'ball' not in kalman_filters:
                        kalman_filters['ball'] = KalmanFilter(
                            F, B, H, Q, R,
                            np.array([pitch_ball_xy[0], pitch_ball_xy[1], 0.0, 0.0]),
                            P0
                        )
                    kf = kalman_filters['ball']
                    kf.predict(u=np.zeros(2))
                    x = kf.update(pitch_ball_xy)
                    pitch_ball_xy = x[:2]
                    ball_history.append((frame_idx, pitch_ball_xy[0], pitch_ball_xy[1], -1))


            
            # Create a base pitch image.
            annotated_frame = draw_pitch(CONFIG)
            if isinstance(pitch_players_xy, (list, np.ndarray)) and len(pitch_players_xy) > 0:
                team0_points = []
                team1_points = []
                for i, pt in enumerate(pitch_players_xy):
                    team_id = players_detections.class_id[i]
                    if team_id == 0:
                        team0_points.append(pt)
                    elif team_id == 1:
                        team1_points.append(pt)
                if team0_points:
                    annotated_frame = draw_points_on_pitch(
                        config=CONFIG,
                        xy=np.array(team0_points),
                        face_color=sv.Color.from_hex('00BFFF'),  # Team 0: Blue
                        edge_color=sv.Color.BLACK,
                        radius=16,
                        pitch=annotated_frame
                    )
                if team1_points:
                    annotated_frame = draw_points_on_pitch(
                        config=CONFIG,
                        xy=np.array(team1_points),
                        face_color=sv.Color.from_hex('FF4500'),  # Team 1: OrangeRed
                        edge_color=sv.Color.BLACK,
                        radius=16,
                        pitch=annotated_frame
                    )
            if isinstance(pitch_referees_xy, (list, np.ndarray)) and len(pitch_referees_xy) > 0:
                            annotated_frame = draw_points_on_pitch(
                                config=CONFIG,
                                xy=np.array(pitch_referees_xy),
                                face_color=sv.Color.from_hex('FF1493'),
                                edge_color=sv.Color.BLACK,
                                radius=16,
                                pitch=annotated_frame
                            )
            if isinstance(pitch_goalkeepers_xy, (list, np.ndarray)) and len(pitch_goalkeepers_xy) > 0:
                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=np.array(pitch_goalkeepers_xy),
                    face_color=sv.Color.from_hex('FFFF00'),  # e.g. yellow for GK
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=annotated_frame
                )
            if pitch_ball_xy is not None and isinstance(pitch_ball_xy, (list, np.ndarray)) and len(np.array(pitch_ball_xy).flatten()) >= 2:
                annotated_frame = draw_points_on_pitch(
                    config=CONFIG,
                    xy=np.array([pitch_ball_xy]),
                    face_color=sv.Color.from_hex('FFFFFF'),
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    pitch=annotated_frame
                )
            annotated_frame = cv2.resize(annotated_frame, (video_info.width, video_info.height))
            videosnk.write_frame(annotated_frame)
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