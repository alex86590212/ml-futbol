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
SOURCE_VIDEO_PATH = "/home2/s5549329/ml-futbol/example_matches/121364_0.mp4"

IMAGES_FOLDER_PATH_PLAYER_DETECTION = "/home2/s5549329/ml-futbol/saved_images/anottated_video.mp4"
IMAGES_FOLDER_PATH_PITCH_DETECTION = "/home2/s5549329/ml-futbol/saved_images/anotatted_pitch_video.mp4"
IMAGES_FOLDER_PATH_PITCH_PLAYER_DETECTION = "/home2/s5549329/ml-futbol/saved_images/anotatted_pitch_player_video.mp4"
IMAGES_FOLDER_BALL_DETECTION = "/home2/s5549329/ml-futbol/saved_images/ball_video.mp4"
IMAGES_FOLDER_PATH = "/home2/s5549329/ml-futbol/saved_images"

FIELD_DETECTION_MODEL = YOLO("/home2/s5549329/ml-futbol/models/downloaded_pitch.pt")
PLAYER_DETECTION_MODEL = YOLO("/home2/s5549329/ml-futbol/models/best_player_100.pt")

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
elif TYPE_OF_VIDEO[NUMBER] == "All_Detection_Video":
    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    print(f"Video width: {video_info.width}, height: {video_info.height}, fps: {video_info.fps}")
    videosnk = sv.VideoSink(IMAGES_FOLDER_PATH_PITCH_PLAYER_DETECTION, video_info, codec="mp4v")
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    #saving the coordinates of each player on 2D pitch per frame

    tracker_history = defaultdict(list)
    last_seen_frame = {}

    prev_pitch_players_xy = None
    prev_pitch_ball_xy = None
    prev_pitch_referees_xy = None
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

            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            if players_crops:  # only predict if there are crops
                players_detections.class_id = team_classifier.predict(players_crops)
            try:
                goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)
            except Exception as e:
                print("Error resolving goalkeeper team IDs:", e)
            referees_detections.class_id -= 1

            merged_detections = sv.Detections.merge([
                players_detections, goalkeepers_detections, referees_detections
            ])

            labels = [f"#{tid}" for tid in merged_detections.tracker_id]
            merged_detections.class_id = merged_detections.class_id.astype(int)
            
            # Use the field detection model to get pitch keypoints
            field_result = FIELD_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            key_points = sv.KeyPoints.from_ultralytics(field_result)
            print(key_points.confidence)

            filter_mask = key_points.confidence[0] > 0.5
            frame_reference_points = key_points.xy[0][filter_mask]
            pitch_reference_points = np.array(CONFIG.vertices)[filter_mask]

            transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )

            frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

            players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_players_xy = transformer.transform_points(points=players_xy)
            
            referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_referees_xy = transformer.transform_points(points=referees_xy)

            for i, tracker_id in enumerate(players_detections.tracker_id):
                team_id = players_detections.class_id[i]
                player_pos = pitch_players_xy[i]
                tracker_history[tracker_id].append((frame_idx, player_pos[0], player_pos[1], team_id))
                last_seen_frame[tracker_id] = frame_idx

            annotated_frame = draw_pitch(CONFIG)
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_xy[players_detections.class_id == 0],
                face_color=sv.Color.from_hex('00BFFF'),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=annotated_frame
            )
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_players_xy[players_detections.class_id == 1],
                face_color=sv.Color.from_hex('FF1493'),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=annotated_frame
            )

            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_ball_xy,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=annotated_frame
            )
            annotated_frame = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_referees_xy,
                face_color=sv.Color.from_hex('FFD700'),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=annotated_frame
            )
            annotated_frame = cv2.resize(annotated_frame, (video_info.width, video_info.height))
            print(f"Annotated frame shape: {annotated_frame.shape}")
            try:
                videosnk.write_frame(annotated_frame)
                print(f"Frame {frame_idx} written successfully.")
            except Exception as e:
                print(f"Error writing frame {frame_idx}: {e}")

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

