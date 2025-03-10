import gymnasium as gym
import os
import numpy as np
import pygame
from gym import spaces

class CustomGridEnv(gym.Env):
    """
    A simple custom grid world environment.
    Grid values:
      0: Empty cell
      1: Wall
      2: Start position
      3: Goal
    """
    pygame.init()

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, grid_layout):
        super().__init__()
        self.grid_layout = np.array(grid_layout)
        self.n_rows, self.n_cols = self.grid_layout.shape
        
        # Action space: 0=Up, 1=Right, 2=Down, 3=Left.
        self.action_space = spaces.Discrete(4)
        # Observation space: Agent's (row, col) position.
        self.observation_space = spaces.Box(low=0, high=max(self.n_rows, self.n_cols),
                                            shape=(2,), dtype=np.int32)
        
        # Find the start position (cell with value 2)
        start_positions = np.argwhere(self.grid_layout == 2)
        if start_positions.shape[0] == 0:
            raise ValueError("No starting position (value 2) found in grid_layout.")
        self.start_pos = tuple(start_positions[0])
        self.agent_pos = self.start_pos

        # Rendering settings.
        self.cell_size = 40  # pixels
        self.screen = None

    def reset(self, seed=None, options=None):
        self.agent_pos = self.start_pos
        return np.array(self.agent_pos), {}

    def step(self, action):
        row, col = self.agent_pos
        # Determine the new position based on action.
        if action == 0:    # Up
            new_row, new_col = row - 1, col
        elif action == 1:  # Right
            new_row, new_col = row, col + 1
        elif action == 2:  # Down
            new_row, new_col = row + 1, col
        elif action == 3:  # Left
            new_row, new_col = row, col - 1
        else:
            new_row, new_col = row, col

        # Check boundaries.
        if new_row < 0 or new_row >= self.n_rows or new_col < 0 or new_col >= self.n_cols:
            new_row, new_col = row, col

        # Check for wall: if cell is a wall (value 1), don’t move.
        if self.grid_layout[new_row, new_col] == 1:
            new_row, new_col = row, col

        self.agent_pos = (new_row, new_col)

        # Check if the goal (value 3) is reached.
        done = self.grid_layout[new_row, new_col] == 3
        reward = 1 if done else -0.1  # Reward shaping: small penalty for each step.

        return np.array(self.agent_pos), reward, done, False, {}

    def render(self, mode=None):
    # Use the environment's render_mode if mode is not provided.
        if mode is None:
            mode = getattr(self, "render_mode", "human")
        width, height = self.n_cols * self.cell_size, self.n_rows * self.cell_size

        if mode == 'human':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((width, height))
                pygame.display.set_caption("Custom Grid Environment")
            self.screen.fill((255, 255, 255))
            # Draw grid cells.
            for r in range(self.n_rows):
                for c in range(self.n_cols):
                    rect = pygame.Rect(c * self.cell_size, r * self.cell_size,
                                    self.cell_size, self.cell_size)
                    if self.grid_layout[r, c] == 1:
                        color = (0, 0, 0)
                    elif self.grid_layout[r, c] == 3:
                        color = (0, 255, 0)
                    else:
                        color = (200, 200, 200)
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)
            # Draw the agent.
            agent_rect = pygame.Rect(self.agent_pos[1] * self.cell_size,
                                    self.agent_pos[0] * self.cell_size,
                                    self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 0, 0), agent_rect)
            pygame.display.flip()
            # Return a frame as a NumPy array.
            return np.array(pygame.surfarray.array3d(self.screen)).transpose(1, 0, 2)

        elif mode == 'rgb_array':
            surface = pygame.Surface((width, height))
            surface.fill((255, 255, 255))
            for r in range(self.n_rows):
                for c in range(self.n_cols):
                    rect = pygame.Rect(c * self.cell_size, r * self.cell_size,
                                    self.cell_size, self.cell_size)
                    if self.grid_layout[r, c] == 1:
                        color = (0, 0, 0)
                    elif self.grid_layout[r, c] == 3:
                        color = (0, 255, 0)
                    else:
                        color = (200, 200, 200)
                    pygame.draw.rect(surface, color, rect)
                    pygame.draw.rect(surface, (255, 255, 255), rect, 1)
            agent_rect = pygame.Rect(self.agent_pos[1] * self.cell_size,
                                    self.agent_pos[0] * self.cell_size,
                                    self.cell_size, self.cell_size)
            pygame.draw.rect(surface, (255, 0, 0), agent_rect)
            return np.array(pygame.surfarray.pixels3d(surface)).transpose(1, 0, 2)
        else:
            super().render(mode=mode)

    def close(self):
        if self.screen is not None:
            pygame.quit()

def record_grid_video(grid_layout, agent_policy, video_folder="videos", video_prefix="grid_video"):
# Create the folder if it doesn't exist.
    os.makedirs(video_folder, exist_ok=True)

    # Instantiate your custom grid environment.
    env = CustomGridEnv(grid_layout)
    env.render_mode = "rgb_array"
    # Wrap the environment with RecordVideo.
    env = gym.wrappers.RecordVideo(env, video_folder=video_folder, name_prefix=video_prefix)

    # Reset the environment.
    obs, _ = env.reset()
    done = False

    # Run one episode.
    while not done:
        # Choose an action using your policy (dummy random policy here).
        action = agent_policy(obs, env.action_space)
        obs, reward, done, truncated, info = env.step(action)
        # It's important that your render('rgb_array') returns a valid frame,
        # which RecordVideo will use to compile the video.
        _ = env.render()
        if done or truncated:
            break

    env.close()
    print(f"Video saved in the '{video_folder}' folder.")

def random_policy(state, action_space):
# Dummy policy: choose a random action.
    return action_space.sample()

if __name__ == "__main__":
    # Define a custom grid layout.
    # Legend: 0 = empty, 1 = wall, 2 = start, 3 = goal.
    grid_layout = [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 0, 3, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]
    
    # Instead of running an interactive simulation loop, call record_grid_video.
    record_grid_video(grid_layout, random_policy)
