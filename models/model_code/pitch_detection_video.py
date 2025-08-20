from tqdm import tqdm
import supervision as sv
import numpy as np
import cv2
import os
from config_models import Config, Models
from auxiliars.view_transformer import ViewTransformer

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

class PitchDetection:
    def __init__(self, config: Config, models: Models, CONFIG):
        self.config = config
        self.models = models
        self.CONFIG = CONFIG

        self.video_info = sv.VideoInfo.from_video_path(self.config.source_video_path)
        self.videosnk = sv.VideoSink(self.config.pitch_detection_path, self.video_info)

        self.edge_annotator = sv.EdgeAnnotator(
            color=sv.Color.from_hex('#00BFFF'),
            thickness=2,
            edges=self.CONFIG.edges
        )
        self.vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex('#FF1493'),
            radius=8
        )
        self.vertex_annotator_2 = sv.VertexAnnotator(
            color=sv.Color.from_hex('#00BFFF'),
            radius=8
        )

    def detect_and_save(self):
        """
        Detects and visualizes the soccer pitch lines and reference points in each frame.

        For each video frame:
            - Detects key pitch points using the YOLO model
            - Computes a transformation between pitch coordinates and frame points
            - Transforms and visualizes the entire pitch wireframe
            - Draws pitch edges, reference points, and transformed vertices
            - Saves the annotated frame to the output video

        Output:
            - Annotated pitch video showing pitch lines and calibration
        """
        frame_generator = sv.get_video_frames_generator(self.config.source_video_path)

        with self.videosnk:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames, desc="video processing"):
                result = self.models.FIELD_DETECTION_MODEL.predict(frame, conf=0.3)[0]
                key_points = sv.KeyPoints.from_ultralytics(result)

                filter = key_points.confidence[0] > 0.5
                frame_reference_points = key_points.xy[0][filter]
                frame_reference_key_points = sv.KeyPoints(
                    xy=frame_reference_points[np.newaxis, ...]
                )

                pitch_reference_points = np.array(self.CONFIG.vertices)[filter]

                transformer = ViewTransformer(
                    source=pitch_reference_points,
                    target=frame_reference_points
                )

                pitch_all_points = np.array(self.CONFIG.vertices)
                frame_all_points = transformer.transform_points(points=pitch_all_points)
                frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

                annotated_frame = frame.copy()
                annotated_frame = self.edge_annotator.annotate(
                    scene=annotated_frame,
                    key_points=frame_all_key_points
                )
                annotated_frame = self.vertex_annotator_2.annotate(
                    scene=annotated_frame,
                    key_points=frame_all_key_points
                )
                annotated_frame = self.vertex_annotator.annotate(
                    scene=annotated_frame,
                    key_points=frame_reference_key_points
                )

                print(annotated_frame.shape)
                self.videosnk.write_frame(annotated_frame)

        print("\n✅ Finished pitch detection and annotation.")