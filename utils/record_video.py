import os
import gymnasium as gym
import numpy as np
from utils.tile_code import discretize_state, extract_features

def record_video(q_table, tile_coder, video_folder="videos", video_prefix="sarsa_tile_video"):
    os.makedirs(video_folder, exist_ok=True)
    env_video = gym.make("MsPacman-v4", render_mode="rgb_array")
    env_video = gym.wrappers.RecordVideo(env_video, video_folder=video_folder, name_prefix=video_prefix)
    state, _ = env_video.reset()
    done = False
    while not done:
        features = extract_features(state)
        tile_indices = discretize_state(features, tile_coder)
        q_vals = np.mean([q_table[idx] for idx in tile_indices], axis=0)
        action = int(np.argmax(q_vals))
        state, reward, terminated, truncated, _ = env_video.step(action)
        done = terminated or truncated
        _ = env_video.render()
    env_video.close()
    print(f"Video saved in {video_folder} with prefix {video_prefix}")