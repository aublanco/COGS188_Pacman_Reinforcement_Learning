import cv2
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

def crop_gameplay_area(state, bottom_crop=30):
    """
    Crops the image to remove UI elements at the bottom.
    """
    return state[:-bottom_crop, :, :]

def extract_pacman_coords(state):
    """
    extracts Pacman's (x,y) coordinates from the game image,
    after cropping out UI elements.
    """
    # Crop the image to remove UI (like lives and score) from the bottom.
    cropped_state = crop_gameplay_area(state, bottom_crop=30)
    
    state_bgr = cv2.cvtColor(cropped_state, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(state_bgr, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for Pacman's color (yellowish).
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphological operations to reduce noise.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Assume the largest contour corresponds to Pacman.
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
            
    height, width, _ = cropped_state.shape
    return (width // 2, height // 2)

def extract_pellet_and_wall_masks(observation, pellet_area_threshold=50):
    """
    Extracts separate binary masks for pellets and walls from the observation.
    Walls and pellets are assumed to share the same RGB value.
    """
    state_bgr = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(state_bgr, cv2.COLOR_BGR2HSV)
    
    # Define the HSV range for walls and pellets.
    lower_bound = np.array([0, 120, 220])
    upper_bound = np.array([10, 255, 255])
    combined_mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the combined mask.
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    pellet_mask = np.zeros_like(combined_mask)
    wall_mask = np.zeros_like(combined_mask)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < pellet_area_threshold:
            # Small contours are considered pellets.
            cv2.drawContours(pellet_mask, [cnt], -1, 255, -1)
        else:
            # Larger contours are treated as walls.
            cv2.drawContours(wall_mask, [cnt], -1, 255, -1)
    
    return pellet_mask, wall_mask

def closest_food_distance(observation, pacman_coords):
    pellet_mask, _ = extract_pellet_and_wall_masks(observation)
    food_pixels = cv2.findNonZero(pellet_mask)
    
    if food_pixels is not None:
        # Compute distances from Pacman's position.
        distances = [np.linalg.norm(np.array(pt[0]) - np.array(pacman_coords)) for pt in food_pixels]
        min_idx = np.argmin(distances)
        closest_coord = tuple(food_pixels[min_idx][0])
        return min(distances), closest_coord
    # If no food is detected, return a high distance.
    return 1000, None
    
if __name__ == "__main__":
    env = gym.make("MsPacman-v4", render_mode="rgb_array")
    state, _ = env.reset()
    
    # Extract Pacman's coordinates.
    pacman_coords = extract_pacman_coords(state)
    print("Pacman Coordinates:", pacman_coords)
    
    # Extract pellet and wall masks.
    pellet_mask, wall_mask = extract_pellet_and_wall_masks(state)
    closest_food, food_coord = closest_food_distance(state, pacman_coords)
    print("Closest food distance:", closest_food)
    print("Closest food coordinates:", food_coord)
    plt.figure(figsize=(12, 4))
    
    # Display original image with Pacman coordinate marked.
    plt.subplot(1, 3, 1)
    state_copy = state.copy()
    cv2.circle(state_copy, pacman_coords, 2, (255, 0, 0), -1)
    if food_coord is not None:
        cv2.circle(state_copy, food_coord, 2, (0, 255, 0), -1)
    plt.imshow(state_copy)
    plt.title("Original Image with Pacman")
    plt.axis("off")
    
    # Display pellet mask.
    plt.subplot(1, 3, 2)
    plt.imshow(pellet_mask, cmap="gray")
    plt.title("Pellet Mask")
    plt.axis("off")
    
    # Display wall mask.
    plt.subplot(1, 3, 3)
    plt.imshow(wall_mask, cmap="gray")
    plt.title("Wall Mask")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("featureExtractor.png")
    plt.show()
    