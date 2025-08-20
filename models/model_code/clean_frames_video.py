from auxiliars.team_classifier import TeamClassifier
from tqdm import tqdm
import supervision as sv
import numpy as np
import pandas as pd
import cv2
import os
from config_models import Config, Models
from collections import deque

import subprocess
import shutil

PLAYER_THRESHOLD = 18
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

class RollingIDs:
    def __init__(self, window=12):  # ~0.5s at 24fps; tune as needed
        self.win = deque(maxlen=window)
    def update(self, ids):
        self.win.append(set(map(int, ids)))
    def unique_count(self):
        s = set()
        for x in self.win:
            s |= x
        return len(s)

class CleanVideo:
    def __init__(self, config: Config, models: Models):
        self.config = config
        self.models = models
        self.player_id = PLAYER_ID
        self.goalkeeper_id = GOALKEEPER_ID
        self.referee_id = REFEREE_ID

        self.video_info = sv.VideoInfo.from_video_path(config.source_video_path)
        self.output_path = config.clean_video_path

        self.total_frames = int(self.video_info.total_frames) if self.video_info.total_frames else 0
        if self.total_frames <= 0:
            cap = cv2.VideoCapture(self.config.source_video_path)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()

        print(f"Video: {self.config.source_video_path}")
        print(f"Frames: {self.total_frames or 'unknown'} | "
              f"FPS: {self.video_info.fps} | "
              f"Size: {self.video_info.width}x{self.video_info.height} | "
              f"Duration: { (self.total_frames / self.video_info.fps):.1f}s" if self.total_frames else "Duration: unknown")


        self.tracker = sv.ByteTrack()
        self.tracker.reset()
        

        self.stats = {
            "player_detected": 0,
            "player_missing": 0,
            "pitch_found": 0,
            "pitch_missing": 0,
            "frames_kept": 0,
            "frames_skipped": 0
        }

        self.codec  = getattr(config, "codec", "libx264")   # or "libx265"
        self.crf    = getattr(config, "crf", 23)            # 18..28 (lower=better)
        self.preset = getattr(config, "preset", "slow")     # ultrafast..placebo

        self._ff = None
        self._cv_writer = None

         # ---- Wide-shot gating params ----
        self.rolling = RollingIDs(window=getattr(config, "players_window", 12))
        # Hysteresis: enter wide at hi, stay wide down to lo
        self.min_players_hi = getattr(config, "min_players_hi", 20)
        self.min_players_lo = getattr(config, "min_players_lo", 16)
        self.state_wide = False

        # Spatial spread (grid occupancy) + scale (median bbox area ratio)
        self.grid_x = getattr(config, "grid_x", 6)
        self.grid_y = getattr(config, "grid_y", 3)
        self.min_cells = getattr(config, "min_cells", 10)  # out of grid_x*grid_y
        self.max_median_area = getattr(config, "max_median_area", 0.018)  # ~1.8% of frame

    # ---------- Writers ----------
    def _start_ffmpeg(self):
        """Start FFmpeg and return a Popen with stdin for raw bgr24 frames."""
        if shutil.which("ffmpeg") is None:
            return None
        w, h = self.video_info.width, self.video_info.height
        fps  = self.video_info.fps
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",                   # stdin raw frames
            "-c:v", self.codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",                       # no audio; add mapping if you need audio
            self.output_path
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)
    
    def _start_cv_writer(self):
        """OpenCV fallback with H.264 if available; may be larger than FFmpeg."""
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # try H.264 tag
        w, h = self.video_info.width, self.video_info.height
        fps  = self.video_info.fps
        writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            # fallback to MP4V (bigger files)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w, h))
        return writer
    
    def _open_writer(self):
        self._ff = self._start_ffmpeg()
        if self._ff is None:
            self._cv_writer = self._start_cv_writer()

    def _write_frame(self, frame: np.ndarray):
        # Ensure correct dtype/layout
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        if self._ff is not None:
            self._ff.stdin.write(frame.tobytes())
        else:
            self._cv_writer.write(frame)

    def _close_writer(self):
        if self._ff is not None:
            try:
                self._ff.stdin.close()
            except Exception:
                pass
            self._ff.wait()
            self._ff = None
        if self._cv_writer is not None:
            self._cv_writer.release()
            self._cv_writer = None

    def detect_and_save(self):
        frame_generator = sv.get_video_frames_generator(self.config.source_video_path)
        self._open_writer()

        for frame_idx, frame in enumerate(tqdm(frame_generator,
                                                total=self.video_info.total_frames,
                                                desc="Detecting Players")):
            # --- PLAYER DETECTION ---
            # result variable is Result object for one frame, it can accept multiple frames, but because we only insert one, we use [0]
            result = self.models.PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            # detections variable is a supervision.Detections object here
            detections = sv.Detections.from_ultralytics(result)
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            print(f"Detections xyxy in {frame_idx} frame:{detections.xyxy}")
            print(f"Detections confidence in {frame_idx} frame: {detections.confidence}")
            print(f"Detections class_id in {frame_idx} frame: {detections.class_id}")

            # tracked variable is still supervision.Detections object here, but with the attribute tracker_id added 
            tracked = self.tracker.update_with_detections(detections)
            print(f"Detections ids in {frame_idx} frame: {tracked.tracker_id}")

            pg_mask = (tracked.class_id == self.player_id) | (tracked.class_id == self.goalkeeper_id)

            # Rolling unique players (by tracker_id)
            if tracked.tracker_id is not None and np.any(pg_mask):
                self.rolling.update(tracked.tracker_id[pg_mask])
            else:
                self.rolling.update([])
            unique_players = self.rolling.unique_count()

            # Spatial spread (grid occupancy) and scale (median bbox area ratio)
            keep_spread, keep_scale = False, False
            if np.any(pg_mask):
                H, W = frame.shape[:2]
                xyxy = tracked.xyxy[pg_mask]
                centers = np.column_stack(((xyxy[:, 0] + xyxy[:, 2]) * 0.5,
                                           (xyxy[:, 1] + xyxy[:, 3]) * 0.5))
                gx, gy = self.grid_x, self.grid_y
                cell_w, cell_h = W / gx, H / gy
                cells = {(int(cx // cell_w), int(cy // cell_h)) for cx, cy in centers}
                keep_spread = (len(cells) >= self.min_cells)

                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                median_area_ratio = float(np.median(areas) / (W * H))
                keep_scale = (median_area_ratio <= self.max_median_area)

            # Hysteresis on player count
            if self.state_wide:
                keep_players = (unique_players >= self.min_players_lo)
            else:
                keep_players = (unique_players >= self.min_players_hi)
            self.state_wide = keep_players

            keep_player = keep_players and keep_spread and keep_scale
            if keep_player:
                self.stats["player_detected"] += 1
            else:
                self.stats["player_missing"] += 1

            # --- PITCH DETECTION ---


            keep_pitch = False
            #field_res variable is a Result object for one frame, it can accept multiple frames, but because we only insert one, we use [0]
            field_res = self.models.FIELD_DETECTION_MODEL.predict(frame, conf=0.3)[0]
            key_pts = sv.KeyPoints.from_ultralytics(field_res)
            #key_pts variables is a supervision.KeyPoints object here

            # 1 pitch instance, 8 keypoints
            #key_pts.confidence = [[0.92, 0.88, 0.12, 0.76, 0.81, 0.55, 0.47, 0.90]]  # shape (1, 8)

            if (key_pts is not None
                    and hasattr(key_pts, "confidence")
                    and key_pts.confidence is not None
                    and len(key_pts.confidence) > 0
                    and len(key_pts.confidence[0]) > 0):
                
                print(f"Pitch xy in {frame_idx} frame:{key_pts.xy}")
                print(f"Pitch confidence in {frame_idx} frame: {key_pts.confidence}")
                mask = key_pts.confidence[0] > 0.5
                #frame_reference_points is numpy array from the supervision.KeyPoints object, but with a mask added of the first instance(there is only an instance)
                frame_reference_points = key_pts.xy[0][mask]

                print(f"Pitch shape in {frame_idx} frame: {frame_reference_points.shape}")
                if frame_reference_points.shape[0] >= 4:
                    keep_pitch = True

            if keep_pitch:
                self.stats["pitch_found"] += 1
            else:
                self.stats["pitch_missing"] += 1

            # --- WRITE FRAME IF BOTH CRITERIA MET ---
            if keep_player and keep_pitch:
                self._write_frame(frame)
                self.stats["frames_kept"] += 1
            else:
                self.stats["frames_skipped"] += 1

        self._close_writer()

        self._log_summary()

    def _log_summary(self):
        total = self.video_info.total_frames
        print("\n--- CleanVideo Filtering Summary ---")
        print(f"Total frames: {total}")
        print(f"Frames with enough players: {self.stats['player_detected']} "
              f"({100 * self.stats['player_detected'] / total:.2f}%)")
        print(f"Frames with enough pitch keypoints: {self.stats['pitch_found']} "
              f"({100 * self.stats['pitch_found'] / total:.2f}%)")
        print(f"Frames kept (both criteria): {self.stats['frames_kept']} "
              f"({100 * self.stats['frames_kept'] / total:.2f}%)")
        print(f"Frames skipped: {self.stats['frames_skipped']} "
              f"({100 * self.stats['frames_skipped'] / total:.2f}%)")
