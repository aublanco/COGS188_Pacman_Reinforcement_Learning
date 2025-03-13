import numpy as np

class TileCoder:
    def __init__(self, n_tilings, n_bins, low, high):
        self.n_tilings = n_tilings
        self.n_bins = np.array(n_bins)
        self.low = np.array(low)
        self.high = np.array(high)
        self.tile_width = (self.high - self.low) / self.n_bins
        # Offsets for each tiling.
        self.offsets = [ (i / self.n_tilings) * self.tile_width for i in range(self.n_tilings) ]
    
    def get_tiles(self, state):
        # state is a numpy array of shape (n_features,)
        scaled_state = (state - self.low) / self.tile_width
        tiles = []
        for offset in self.offsets:
            tile_indices = tuple(np.floor(scaled_state + offset).astype(int))
            tiles.append(tile_indices)
        return tiles
    
    def num_tiles(self):
        return self.n_tilings * np.prod(self.n_bins)
    
def create_q_table(n_actions, tile_coder):
    return np.zeros((tile_coder.num_tiles(), n_actions))

def get_tile_indices(tile_coder, tiles):
    indices = []
    n_bins = np.array(tile_coder.n_bins)
    for i, tile in enumerate(tiles):
        # Clip tile indices to be within valid range.
        tile = np.clip(tile, 0, n_bins - 1)
        index = i * np.prod(n_bins) + np.ravel_multi_index(tile, n_bins)
        indices.append(index)
    return indices

def get_q_values(q_table, tile_indices):
    return np.mean([q_table[idx] for idx in tile_indices], axis=0)

def discretize_state(state, tile_coder):
    tiles = tile_coder.get_tiles(state)
    tile_indices = get_tile_indices(tile_coder, tiles)
    return tile_indices


def extract_features(state):
    # Use the first three channels (RGB) to compute the grayscale image.
    gray = np.dot(state[..., :3], [0.2989, 0.5870, 0.1140])
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    return np.array([mean_val, std_val])