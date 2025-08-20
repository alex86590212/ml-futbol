import pandas as pd
import gym
import numpy as np

players = pd.read_csv("ml-futbol/frames_player_ball/player_coordinates.csv")
ball = pd.read_csv("ml-futbol/frames_player_ball/ball_coordinates.csv")
possesion = pd.read_csv("ml-futbol/frames_player_ball/team_possession.csv")

FIELD_WIDTH = 12000
FIELD_HEIGHT = 7000
MAX_STEP = 100  # max movement per frame


def build_frame_dict():
    frames = {}
    possession_indexed = possession.set_index("frame_idx")

    for frame_id in sorted(players["frame_idx"].unique()):
        frame_players = players[players["frame_idx"] == frame_id]
        ball_row = ball[ball["frame_idx"] == frame_id].iloc[0]
        poss_row = possession_indexed.loc[frame_id]
        ball_vel = ball_velocity[ball_velocity["frame_idx"] == frame_id]

        # Merge velocities into player records
        vel_dict = player_velocity[player_velocity["frame_idx"] == frame_id].set_index("tracker_id")
        player_dicts = []
        for _, row in frame_players.iterrows():
            tracker_id = row["tracker_id"]
            vx, vy = vel_dict.loc[tracker_id][["vx", "vy"]] if tracker_id in vel_dict.index else (0.0, 0.0)
            player_dicts.append({
                "tracker_id": tracker_id,
                "x": row["x"],
                "y": row["y"],
                "vx": vx,
                "vy": vy,
                "team_id": row["team_id"],
                "has_ball": row["has_ball"]
            })

        frames[frame_id] = {
            "players": player_dicts,
            "ball": {
                "x": ball_row["x"],
                "y": ball_row["y"],
                "vx": ball_vel["vx"].iloc[0] if not ball_vel.empty else 0.0,
                "vy": ball_vel["vy"].iloc[0] if not ball_vel.empty else 0.0
            },
            "possession": {
                "team_0": poss_row["team_0_possession"],
                "team_1": poss_row["team_1_possession"]
            },
            "centroid": {
                "team_0_centroid_x": poss_row["team_0_centroid_x"],
                "team_0_centroid_y": poss_row["team_0_centroid_y"],
                "team_0_spread": poss_row["team_0_spread"],
                "team_1_centroid_x": poss_row["team_1_centroid_x"],
                "team_1_centroid_y": poss_row["team_1_centroid_y"],
                "team_1_spread": poss_row["team_1_spread"]
            }
        }
    return frames

frame_data = build_frame_dict()

class FotballTeamEnv(gym.Env):
    def __init__(self, frame_data, num_players=11, team_home_id=0, other_team_id=1):
        super().__init__()
        self.fram_data = frame_data
        self.frame_ids = sorted(list(frame_data.keys()))
        self.num_players = num_players

        self.team_home_id = team_home_id
        self.other_team_id = other_team_id
        self.controlled_positions = None

        obs_size = 4 + 2 * num_players * 5 + 6
        self.action_space = gym.spaces.Box(low= -1.0, high= 1.0, shape=(num_players, 2), dtype=np.float32)

        #each player sees the ball 2 , 11 players 3, 11 players 3
        self.observation_space = (ow=0, high=7000, shape=(obs_size, dtype=np.float32))
        self.reset()

    def reset(self):
        self.current_idx = 0
        self.frame_id = self.frame_ids[self.current_idx]
        self.controlled_positions = None
        return self._get_obs()

    def step(self, action):
        self._apply_action(action)  
        self._advance_frame()       # move to next frame 

        reward = self._calculate_reward()
        obs = self._get_obs()
        done = self.current_idx >= len(self.frame_ids) - 1
        return obs, reward, done, {}

    
    def _apply_action(self, action):
        frame = self.frame_data[self.frame_id]
        home_players = [p for p in frame["players"] if p["team_id"] == self.team_home_id]
        home_players = sorted(home_players, key=lambda p: p["tracker_id"])[:self.num_players]
        self.controlled_positions = []

        for i, (dx, dy) in enumerate(action):
            base = home_players[i]
            new_x = np.clip(base["x"] + dx * MAX_STEP, 0, FIELD_WIDTH)
            new_y = np.clip(base["y"] + dy * MAX_STEP, 0, FIELD_HEIGHT)
            self.controlled_positions.append({
                "x": new_x,
                "y": new_y,
                "vx": dx * MAX_STEP,
                "vy": dy * MAX_STEP,
                "has_ball": base["has_ball"],
                "tracker_id": base["tracker_id"]
            })

    def _advance_frame(self):
        self.current_idx += 1
        if self.current_idx < len(self.frame_ids):
            self.frame_id = self.frame_ids[self.current_idx]

    def _calculate_reward(self):
         frame = self.frame_data[self.frame_id]
        if frame["possession"][f"team_{self.team_home_id}"]:
            return 1.0
        elif frame["possession"][f"team_{self.other_team_id}"]:
            return -1.0
        return 0.0


    def _get_obs(self):
        frame = self.frame_data[self.frame_id]
        obs = []

        # Ball (x, y, vx, vy)
        obs += [frame["ball"]["x"], frame["ball"]["y"], frame["ball"]["vx"], frame["ball"]["vy"]]

        # Controlled team (home) - use simulated positions if available
        home_players = sorted([p for p in frame["players"] if p["team_id"] == self.team_home_id], key=lambda p: p["tracker_id"])[:self.num_players]
        for i in range(self.num_players):
            if self.controlled_positions:
                p = self.controlled_positions[i]
                obs += [p["x"], p["y"], p["vx"], p["vy"], int(p["has_ball"])]
            else:
                p = home_players[i]
                obs += [p["x"], p["y"], p["vx"], p["vy"], int(p["has_ball"])]

        # Opponent team
        opp_players = sorted([p for p in frame["players"] if p["team_id"] == self.other_team_id], key=lambda p: p["tracker_id"])[:self.num_players]
        for p in opp_players:
            obs += [p["x"], p["y"], p["vx"], p["vy"], int(p["has_ball"])]

        # Centroids and spread
        obs += [
            frame["centroid"]["team_0_centroid_x"], frame["centroid"]["team_0_centroid_y"], frame["centroid"]["team_0_spread"],
            frame["centroid"]["team_1_centroid_x"], frame["centroid"]["team_1_centroid_y"], frame["centroid"]["team_1_spread"]
        ]

    
