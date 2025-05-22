from tqdm import tqdm
import supervision as sv
import numpy as np
import pandas as pd
import cv2
import os
from collections import defaultdict
from auxiliars.view_transformer import ViewTransformer
from auxiliars.kalman_filter import KalmanFilter
from auxiliars.pitch_draw import draw_pitch, draw_points_on_pitch
from auxiliars.team_classifier import TeamClassifier
from auxiliars.pitch_config import SoccerPitchConfiguration
from config_models import Config, Models

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
                self.ball_kf = KalmanFilter(
                    self.F, self.B, self.H, self.Q, self.R,
                    x0=np.array([pitch_ball_xy[0], pitch_ball_xy[1], 0., 0.]),
                    P0=self.P0
                )
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

        # Save ball coordinates
        ball_coords = [{"frame_idx": frame_idx, "x": x, "y": y} for frame_idx, x, y in self.ball_history]
        pd.DataFrame(ball_coords).to_csv(os.path.join(output_folder, "ball_coordinates.csv"), index=False)
        print("✅ Ball coordinates log saved.")

        player_records = []
        proximity = defaultdict(list)

        # Create lookup of ball positions
        ball_lookup = {f: (x, y) for f, x, y in self.ball_history}
        frame_to_possession = {}

        for frame_idx in range(self.video_info.total_frames):
            players_in_frame = []

            for tracker_id, records in self.tracker_history.items():
                record_dict = {rec[0]: rec for rec in records}
                if frame_idx in record_dict:
                    _, x, y, team_id = record_dict[frame_idx]
                    dist = float('inf')
                    if frame_idx in ball_lookup:
                        bx, by = ball_lookup[frame_idx]
                        dist = ((x - bx) ** 2 + (y - by) ** 2) ** 0.5
                    players_in_frame.append({
                        "tracker_id": tracker_id,
                        "x": x,
                        "y": y,
                        "team_id": team_id,
                        "distance_to_ball": dist
                    })

            # Determine who has possession
            player_with_ball = min(players_in_frame, key=lambda p: p["distance_to_ball"]) if players_in_frame else None
            possession_team = player_with_ball["team_id"] if player_with_ball and player_with_ball["distance_to_ball"] < 3.0 else -1  # Threshold in pitch units

            for player in players_in_frame:
                player_records.append({
                    "frame_idx": frame_idx,
                    "tracker_id": player["tracker_id"],
                    "x": player["x"],
                    "y": player["y"],
                    "team_id": player["team_id"],
                    "has_ball": player["tracker_id"] == player_with_ball["tracker_id"] if player_with_ball else False,
                    "distance_to_ball": player["distance_to_ball"]
                })

            frame_to_possession[frame_idx] = {
                "frame_idx": frame_idx,
                "team_0_possession": possession_team == 0,
                "team_1_possession": possession_team == 1
            }

        # Save player coordinates + possession
        pd.DataFrame(player_records).to_csv(os.path.join(output_folder, "player_coordinates.csv"), index=False)
        print("✅ Player coordinates (with possession) log saved.")

        # Save team possession
        pd.DataFrame.from_dict(frame_to_possession, orient="index").to_csv(
            os.path.join(output_folder, "team_possession.csv"), index=False)
        print("✅ Team possession log saved.")