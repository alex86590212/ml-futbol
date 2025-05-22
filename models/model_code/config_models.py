from ultralytics import YOLO

class Config:
    def __init__(self, NUMBER):
        self.number = NUMBER
        self.kind_of_videos =  {0: "None", 1: "Player_Detection_Video", 2: "Pitch_Detection_Video", 3: "All_Detection_Video", 4: "Anotatted_Ball_Detection"}
        self.type_of_video = self.kind_of_videos[self.number]

        self.source_video_path = "ml-futbol/example_matches/0bfacc_0.mp4"
        self.images_folder_path = "ml-futbol/saved_images"
        self.pitch_detection_path = "ml-futbol/saved_images/anotatted_pitch_video.mp4"
        self.player_detection_path = "ml-futbol/saved_images/anottated_player_video.mp4"
        self.ball_detection_path = "ml-futbol/saved_images/ball_video_detection.mp4"
        self.homographic_detection_path = "ml-futbol/saved_images/anotatted_pitch_player_video.mp4"

class Models:
    def __init__(self):
        self.FIELD_DETECTION_MODEL = YOLO("ml-futbol/models/downloaded_pitch.pt")
        self.PLAYER_DETECTION_MODEL = YOLO("ml-futbol/models/football-player-detection-v9.pt")
        self.BALL_DETECTION_MODEL = YOLO("ml-futbol/models/football-ball-detection-v2.pt")
