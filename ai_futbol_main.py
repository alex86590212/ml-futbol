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
from umap.umap_ import UMAP
from sklearn.cluster import KMeans
from team_classifier import TeamClassifier

#os.makedirs("example_matches", exist_ok=True)
#url = "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
#url_path = "example_matches/0bfacc_0.mp4"
#gdown.download(url, url_path, quiet=False)

IMAGES_FOLDER_PATH = "/home2/s5549329/ml-futbol/saved_images/anottated_video.mp4"

PLAYER_DETECTION_MODEL = YOLO("/home2/s5549329/ml-futbol/models/best.pt")

SOURCE_VIDEO_PATH = "/home2/s5549329/ml-futbol/example_matches/0bfacc_0.mp4"
print(os.path.exists(SOURCE_VIDEO_PATH))

# saving the first frame of the video
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame = next(frame_generator)

#frame_path = os.path.join(IMAGES_FOLDER_PATH, f"frame_{0:03d}.jpg")
#cv2.imwrite(frame_path, frame)

#anntotations for the players
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
STRIDE = 30

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


#Dividing the players into two systems by using Kmeans clustering


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


# The main part where the video is being produced
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
videosnk = sv.VideoSink(IMAGES_FOLDER_PATH, video_info)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

with videosnk:
    for frame in tqdm(frame_generator, total=video_info.total_frames, desc="video processing"):
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
        videosnk.write_frame(annotated_frame)

