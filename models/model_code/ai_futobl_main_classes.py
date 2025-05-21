from ultralytics import YOLO
from team_classifier import TeamClassifier
from tqdm import tqdm
import supervision as sv
from collections import defaultdict
import numpy as np
from pitch_draw import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch
from pitch_config import SoccerPitchConfiguration
from view_transformer import ViewTransformer
from kalman_filter import KalmanFilter
import pandas as pd
import cv2
import os
import hydra
from omegaconf import DictConfig, OmegaConf


def interpolate_and_extrapolate_ball_history(ball_history, total_frames, max_interp_gap=10, max_extrap_gap=10, decay=0.85):
    """
    Interpolates and extrapolates ball positions across video frames.

    This function fills in missing ball positions by linearly interpolating between
    detected points and extrapolating forward when the ball is temporarily undetected.

    Args:
        ball_history (list): List of tuples (frame_idx, x, y) with known ball positions.
        total_frames (int): Total number of frames in the video.
        max_interp_gap (int): Maximum allowed frame gap for interpolation.
        max_extrap_gap (int): Maximum number of frames allowed for extrapolation.
        decay (float): Velocity decay factor applied during extrapolation.

    Returns:
        dict: A dictionary mapping frame_idx to (x, y) positions, including interpolated or extrapolated values.
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
    """
    Interpolates missing player positions for each tracker ID across all video frames.

    For each tracker, if a position is missing at a given frame, it estimates the position
    using linear interpolation between the nearest known previous and next positions.

    Args:
        tracker_history (dict): Dictionary mapping tracker IDs to a list of (frame_idx, x, y, team_id).
        total_frames (int): Total number of frames in the video.

    Returns:
        defaultdict: Dictionary mapping tracker_id to a list of interpolated records for each frame.
    """ 
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

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

def resolve_goalkeepers_team_id(players: sv.Detections, goalkeepers: sv.Detections) -> np.ndarray:
    """
    Assigns team IDs to goalkeepers based on their proximity to team player centroids.

    The method assumes each team has a spatial center (centroid), and a goalkeeper is
    classified as belonging to the team whose centroid is closest to them.

    Args:
        players (sv.Detections): Detections of all outfield players, with class_id indicating team.
        goalkeepers (sv.Detections): Detections of goalkeepers (no class/team label yet).

    Returns:
        np.ndarray: Array of assigned team IDs for each goalkeeper (0 or 1).
    """
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

class Config:
    def __init__(self, NUMBER):
        self.number = NUMBER
        self.kind_of_videos =  {0: "None", 1: "Player_Detection_Video", 2: "Pitch_Detection_Video", 3: "All_Detection_Video", 4: "Anotatted_Ball_Detection"}
        self.type_of_video = self.kind_of_videos[self.number]

        self.source_video_path = "/home2/s5549329/ml-futbol/example_matches/0bfacc_0.mp4"
        self.images_folder_path = "/home2/s5549329/ml-futbol/saved_images"
        self.pitch_detection_path = "/home2/s5549329/ml-futbol/saved_images/anotatted_pitch_video.mp4"
        self.player_detection_path = "/home2/s5549329/ml-futbol/saved_images/anottated_player_video.mp4"
        self.ball_detection_path = "/home2/s5549329/ml-futbol/saved_images/ball_video_detection.mp4"
        self.homographic_detection_path = "/home2/s5549329/ml-futbol/saved_images/anotatted_pitch_player_video.mp4"


class Models:
    def __init__(self):
        self.FIELD_DETECTION_MODEL = YOLO("/home2/s5549329/ml-futbol/models/downloaded_pitch.pt")
        self.PLAYER_DETECTION_MODEL = YOLO("/home2/s5549329/ml-futbol/models/football-player-detection-v9.pt")
        self.BALL_DETECTION_MODEL = YOLO("/home2/s5549329/ml-futbol/models/football-ball-detection-v2.pt")


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


class PlayerDetection:
    def __init__(self, config: Config, models: Models, team_classifier: TeamClassifier):
        self.config = config
        self.models = models
        self.team_classifier = team_classifier
        self.player_id = PLAYER_ID
        self.goalkeeper_id = GOALKEEPER_ID
        self.referee_id = REFEREE_ID

        self.video_info = sv.VideoInfo.from_video_path(config.source_video_path)
        self.output_path = config.player_detection_path

        # Annotators
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER
        )

        self.tracker = sv.ByteTrack()
        self.tracker.reset()

        self.stats = {
            "player_detected": 0,
            "player_missing": 0
        }

    def detect_and_save(self):
        """
        Detects, classifies, tracks, and annotates all players, goalkeepers, and referees.

        For each frame:
            - Uses YOLO to detect player-related classes
            - Applies ByteTrack to maintain identity tracking
            - Crops player images and uses the team classifier to assign team IDs
            - Resolves goalkeepers’ teams by proximity to team centroids
            - Normalizes referee labels to a common ID
            - Annotates each detection with ellipses and ID labels
            - Saves annotated frame to video

        Output:
            - Annotated player detection video
            - Printed detection summary
        """
        frame_generator = sv.get_video_frames_generator(self.config.source_video_path)
        video_sink = sv.VideoSink(self.output_path, self.video_info)

        with video_sink:
            for frame_idx, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames, desc="Detecting Players")):
                result = self.models.PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
                detections = sv.Detections.from_ultralytics(result)
                detections = detections.with_nms(threshold=0.5, class_agnostic=True)

                # Tracking
                tracked = self.tracker.update_with_detections(detections)

                # Separate classes
                players = tracked[tracked.class_id == self.player_id]
                goalkeepers = tracked[tracked.class_id == self.goalkeeper_id]
                referees = tracked[tracked.class_id == self.referee_id]

                if len(players) > 0:
                    self.stats["player_detected"] += 1
                else:
                    self.stats["player_missing"] += 1

                # Classify players into teams
                player_crops = [sv.crop_image(frame, box) for box in players.xyxy]
                if player_crops:
                    preds = self.team_classifier.predict(player_crops)
                    if len(preds) == len(players):
                        players.class_id = preds
                    else:
                        print(f"⚠️ Mismatch: {len(preds)} labels vs {len(players)} detections")

                # Assign goalkeepers to teams
                goalkeepers.class_id = resolve_goalkeepers_team_id(players, goalkeepers)

                # Normalize referees (class_id = -1)
                referees.class_id -= 1

                # Merge and annotate
                all_detections = sv.Detections.merge([players, goalkeepers, referees])
                labels = [f"#{tid}" for tid in all_detections.tracker_id]
                all_detections.class_id = all_detections.class_id.astype(int)

                annotated = frame.copy()
                annotated = self.ellipse_annotator.annotate(scene=annotated, detections=all_detections)
                annotated = self.label_annotator.annotate(scene=annotated, detections=all_detections, labels=labels)

                video_sink.write_frame(annotated)

        self._log_summary()

    def _log_summary(self):
        """
        Logs the number of frames where players were detected vs. missed.

        Output:
            - Count and percentage of player-detected frames
            - Count and percentage of missed frames
        """
        total = self.video_info.total_frames
        detected = self.stats["player_detected"]
        missing = self.stats["player_missing"]
        print("\n--- Player Detection Summary ---")
        print(f"Total frames: {total}")
        print(f"Players detected in: {detected} frames ({100 * detected / total:.2f}%)")
        print(f"No players detected in: {missing} frames ({100 * missing / total:.2f}%)")


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

class AllDetectionVideo:
    def __init__(self, config: Config, models: Models, team_classifier: TeamClassifier, CONFIG, kalman_params):
        self.config = config
        self.models = models
        self.team_classifier = team_classifier
        self.CONFIG = CONFIG

        self.video_info = sv.VideoInfo.from_video_path(config.source_video_path)
        self.output_path = config.homographic_detection_path

        self.BALL_ID = BALL_ID
        self.GOALKEEPER_ID = GOALKEEPER_ID
        self.PLAYER_ID = PLAYER_ID
        self.REFEREE_ID = REFEREE_ID

        dt = 1.0 / self.video_info.fps
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1,  0], [0, 0, 0,  1]])
        self.B = kalman_params["B"]
        self.H = kalman_params["H"]
        self.Q = kalman_params["Q"]
        self.R = kalman_params["R"]
        self.P0 = kalman_params["P0"]

        self.tracker = sv.ByteTrack()
        self.tracker.reset()

        self.kalman_filters = {}
        self.ball_kf = None
        self.tracker_history = defaultdict(list)
        self.ball_history = []
        self.player_visibility = defaultdict(list)
        self.ball_visibility = []
       
    def detect_and_interpolate(self):
        """
        Runs complete player, ball, and pitch detection, applies Kalman filtering, and stores all tracked data.

        Per frame:
            - Detects and classifies players, goalkeepers, and referees
            - Detects pitch keypoints and computes view transformation
            - Tracks and smooths player and ball positions using Kalman Filters
            - Stores positions and visibility in history logs
            - Handles cases where pitch or ball data is unavailable

        After processing:
            - Interpolates and extrapolates missing ball and player positions
            - Prepares full trajectories for rendering and analysis
        """
        frame_generator = sv.get_video_frames_generator(self.config.source_video_path)

        for frame_idx, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames, desc="Processing video")):
            if frame is None:
                print(f"⚠️ Frame {frame_idx} is empty.")
                continue

            print(f"📹 Frame {frame_idx}:")

            # --- PLAYER & REFEREE DETECTION ---
            result = self.models.PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result).with_nms(threshold=0.5, class_agnostic=True)
            tracked = self.tracker.update_with_detections(detections)

            players = tracked[tracked.class_id == self.PLAYER_ID]
            goalkeepers = tracked[tracked.class_id == self.GOALKEEPER_ID]
            referees = tracked[tracked.class_id == self.REFEREE_ID]

            #Track player visibility
            detected_ids = set(players.tracker_id if players else [])
            all_ids = set(self.player_visibility.keys()).union(detected_ids)
            for tid in all_ids:
                self.player_visibility[tid].append(tid in detected_ids)

            # Team classification
            player_crops = [sv.crop_image(frame, bb) for bb in players.xyxy]
            if player_crops:
                try:
                    preds = self.team_classifier.predict(player_crops)
                    if len(preds) == len(players):
                        players.class_id = preds
                except Exception as e:
                    print("⚠️ Team classifier failed:", e)

            # Normalize referee class ID
            referees.class_id = np.full_like(referees.class_id, -1)
            goalkeepers.class_id = resolve_goalkeepers_team_id(players, goalkeepers)

            # --- BALL DETECTION ---
            ball_result = self.models.BALL_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            ball_detections = sv.Detections.from_ultralytics(ball_result)
            ball_detections = ball_detections[ball_detections.class_id == self.BALL_ID]
            ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10)

            #Track ball visibility
            ball_detected = ball_detections is not None and len(ball_detections) > 0
            self.ball_visibility.append(ball_detected)
            print(f"{'✅' if ball_detected else '❌'} Ball detected: {len(ball_detections) if ball_detected else 0}")

            # --- PITCH DETECTION ---
            field_result = self.models.FIELD_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            keypoints = sv.KeyPoints.from_ultralytics(field_result)
            mask = keypoints.confidence[0] > 0.5
            frame_reference_points = keypoints.xy[0][mask]
            pitch_reference_points = np.array(self.CONFIG.vertices)[mask]

            if len(frame_reference_points) < 4:
                print("⚠️ Not enough pitch points for homography.")
                continue

            transformer = ViewTransformer(
                source=frame_reference_points,
                target=pitch_reference_points
            )

            # --- TRACK BALL ---
            if ball_detected:
                ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_ball_xy = transformer.transform_points(ball_xy)[0]
                if self.ball_kf is None:
                    self.ball_kf = KalmanFilter(self.F, self.B, self.H, self.Q, self.R,
                                                x0=np.array([pitch_ball_xy[0], pitch_ball_xy[1], 0., 0.]), P0=self.P0)
                self.ball_kf.predict(u=np.zeros(2))
                x = self.ball_kf.update(pitch_ball_xy)
                self.ball_history.append((frame_idx, x[0], x[1]))
            elif self.ball_kf:
                self.ball_kf.predict(u=np.zeros(2))
                x = self.ball_kf.x
                self.ball_history.append((frame_idx, x[0], x[1]))

            # --- TRACK PLAYERS ---
            players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_players_xy = transformer.transform_points(players_xy)

            for i, tracker_id in enumerate(players.tracker_id):
                team_id = players.class_id[i]
                z = pitch_players_xy[i].flatten()
                if tracker_id not in self.kalman_filters:
                    self.kalman_filters[tracker_id] = KalmanFilter(self.F, self.B, self.H, self.Q, self.R,
                                                                   x0=np.array([z[0], z[1], 0., 0.]), P0=self.P0)
                kf = self.kalman_filters[tracker_id]
                kf.predict(u=np.zeros(2))
                x = kf.update(z)
                self.tracker_history[tracker_id].append((frame_idx, x[0], x[1], team_id))

        self.interpolated_ball_history = interpolate_and_extrapolate_ball_history(
            ball_history=self.ball_history,
            total_frames=self.video_info.total_frames,
            max_interp_gap=10,
            max_extrap_gap=10,
            decay=0.85
        )

        self.interpolated_tracker_history = interpolate_players_tracker_history(
            tracker_history=self.tracker_history,
            total_frames=self.video_info.total_frames
        )
        
        # Makes sure all player visibility lists are the same length as the number of video frames by padding with False.
        max_len = self.video_info.total_frames
        for tid, visibility in self.player_visibility.items():
            if len(visibility) < max_len:
                self.player_visibility[tid] += [False] * (max_len - len(visibility))

    def render_interpolated_video(self):
        """
        Renders a new video that visualizes interpolated player and ball trajectories on a synthetic pitch.

        - Draws pitch wireframe
        - Overlays team player positions with different colors
        - Marks interpolated ball positions
        - Writes output video frame-by-frame at native resolution

        Output:
            - Annotated video showing all player and ball trajectories (real and interpolated)
        """
        # This structure groups all interpolated player positions by frame, which makes it easier to draw them frame-by-frame later.
        # frame_player_map is a nested dictionary: {frame_idx: {tracker_id: (x, y, team_id)}}
        frame_player_map = defaultdict(dict)
        for tid, records in self.interpolated_tracker_history.items():
            for frame_idx, x, y, team_id in records:
                frame_player_map[frame_idx][tid] = (x, y, team_id)

        frame_generator = sv.get_video_frames_generator(self.config.source_video_path)
        video_sink = sv.VideoSink(self.output_path, self.video_info, codec="mp4v")

        with video_sink:
            for frame_idx, frame in enumerate(tqdm(frame_generator, total=self.video_info.total_frames, desc="Rendering combined detected + interpolated frames")):
                annotated = draw_pitch(self.CONFIG)

                team0_pts, team1_pts = [], []
                # frame_player_map is a nested dictionary: {frame_idx: {tracker_id: (x, y, team_id)}}
                for tid, (x, y, team_id) in frame_player_map[frame_idx].items():
                    # Checks if this player (tid) was actually detected in this frame.
                    # If the player has never been detected before (not in self.player_visibility),
                    # we default to a list of all False (i.e., never visible in any frame).
                    if self.player_visibility.get(tid, [False] * self.video_info.total_frames)[frame_idx]:
                        # Creates a dictionary of real (Kalman-filtered) positions for this player.
                        # rec[0] = frame index, and rec[1:] = x, y, team_id
                        #real_frames = {
                            #frame_idx: (frame_idx, x, y, team_id),
                        real_frames = {rec[0]: rec for rec in self.tracker_history.get(tid, [])}
                        if frame_idx in real_frames and None not in real_frames[frame_idx][1:3]:
                            x, y, team_id = real_frames[frame_idx][1:]
                    if team_id == 0:
                        team0_pts.append((x, y))
                    elif team_id == 1:
                        team1_pts.append((x, y))

                if team0_pts:
                    annotated = draw_points_on_pitch(
                        config=self.CONFIG,
                        xy=np.array(team0_pts),
                        face_color=sv.Color.from_hex('00BFFF'),
                        edge_color=sv.Color.BLACK,
                        radius=16,
                        pitch=annotated
                    )
                if team1_pts:
                    annotated = draw_points_on_pitch(
                        config=self.CONFIG,
                        xy=np.array(team1_pts),
                        face_color=sv.Color.from_hex('FF1493'),
                        edge_color=sv.Color.BLACK,
                        radius=16,
                        pitch=annotated
                    )

                if frame_idx in self.interpolated_ball_history:
                    x, y = self.interpolated_ball_history[frame_idx]
                    annotated = draw_points_on_pitch(
                        config=self.CONFIG,
                        xy=np.array([[x, y]]),
                        face_color=sv.Color.from_hex('FFFFFF'),
                        edge_color=sv.Color.BLACK,
                        radius=10,
                        pitch=annotated
                    )

                annotated = cv2.resize(annotated, (self.video_info.width, self.video_info.height))
                video_sink.write_frame(annotated)

        print("\n✅ All Detection Video rendered and saved with interpolated frames.")

    def export_tracking_tables(self, output_folder: str):
        """
        Exports all trajectory and visibility data into CSV logs for further analysis.

        Files created:
            - `player_visibility.csv`: Binary matrix of player visibility per frame
            - `ball_visibility.csv`: Whether the ball was detected in each frame
            - `player_coordinates.csv`: Kalman-filtered positions per player ID per frame

        Args:
            output_folder (str): Directory to save all exported CSV files
        """
        os.makedirs(output_folder, exist_ok=True)

        # Save player visibility
        player_df = pd.DataFrame(self.player_visibility).fillna(False).astype(bool)
        player_df.index.name = "frame_idx"
        player_df.to_csv(os.path.join(output_folder, "player_visibility.csv"))
        print("✅ Player visibility log saved.")

        # Save ball visibility
        pd.DataFrame({
            "frame_idx": list(range(len(self.ball_visibility))),
            "ball_visible": self.ball_visibility
        }).to_csv(os.path.join(output_folder, "ball_visibility.csv"), index=False)
        print("✅ Ball visibility log saved.")

        # Save player coordinates
        player_coords = []
        for tracker_id, records in self.tracker_history.items():
            for frame_idx, x, y, team_id in records:
                player_coords.append({
                    "frame_idx": frame_idx,
                    "tracker_id": tracker_id,
                    "x": x,
                    "y": y,
                    "team_id": team_id
                })
        pd.DataFrame(player_coords).to_csv(os.path.join(output_folder, "player_coordinates.csv"), index=False)
        print("✅ Player coordinates log saved.")

@hydra.main(config_path="configs", config_name="kalman")
def main(cfg: DictConfig):
    # Initialize config, models, and pitch configuration
    config = Config(NUMBER=3)  # 3 corresponds to "All_Detection_Video"
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
        output_folder = "/home2/s5549329/ml-futbol/frames_player_ball"
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