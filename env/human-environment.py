import gymnasium as gym
import pygame
import numpy as np
import ale_py

def main():
    # Initialize the Gym environment with rgb_array rendering
    gym.register_envs(ale_py)
    env = gym.make("MsPacmanDeterministic-v4", render_mode="rgb_array")
    env = env.unwrapped
    observation, info = env.reset()

    # Initialize Pygame and create a display window matching the environment's frame size
    pygame.init()
    height, width, _ = observation.shape
    scaled_width, scaled_height = width * 3, height * 2
    screen = pygame.display.set_mode((scaled_width, scaled_height))
    pygame.display.set_caption("Play Ms. Pacman")
    clock = pygame.time.Clock()

    # Define a mapping from pygame keys to environment actions.
    # Typically, for the minimal action set of Ms. Pacman:
    # 0: NOOP, 1: UP, 2: RIGHT, 3: LEFT, 4: DOWN.
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_RIGHT: 2,
        pygame.K_LEFT: 3,
        pygame.K_DOWN: 4,
    }
    action = 0  # default to NOOP

    done = False
    while not done:
        # Process Pygame events for keyboard input and window close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action = key_to_action[event.key]
            elif event.type == pygame.KEYUP:
                # When the key is released, revert to NOOP (customize as needed)
                action = 0

        # Step the environment. Note that Gymnasium returns terminated and truncated flags.
        observation, reward, terminated, truncated, info = env.step(action)
        
        # An episode ends if either terminated or truncated is True.
        done = terminated or truncated

        # Poll keys for more immediate action responses
        keys = pygame.key.get_pressed()
        action = 0  # default action (NOOP)
        if keys[pygame.K_UP]:
            action = 1
        elif keys[pygame.K_RIGHT]:
            action = 2
        elif keys[pygame.K_LEFT]:
            action = 3
        elif keys[pygame.K_DOWN]:
            action = 4

        # Get the frame from the environment (as an RGB numpy array)
        frame = env.render()

        # Convert the frame for proper orientation in Pygame.
        # Note: Depending on the version, you might need to adjust the transformation.
        frame = np.transpose(frame, (1, 0, 2))
        frame_surface = pygame.surfarray.make_surface(frame)
        frame_surface = pygame.transform.scale(frame_surface, (scaled_width, scaled_height))

        # Draw the frame on the screen and update display
        screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

        # Limit the loop to 30 frames per second
        clock.tick(30)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()
