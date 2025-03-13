def compute_shaped_reward(original_reward, discrete_state, pellet_status, bonus=10.0, penalty=-1.0):
    """
    Computes a shaped reward based on whether the discretized state has been visited before.
    If the state is visited for the first time, a bonus is added (pellets not collected).
    Otherwise, a penalty is added (pellets already collected).

    Args:
        original_reward (float): The reward from the environment.
        discrete_state (tuple): The discrete representation of the state (e.g., tile indices tuple).
        pellet_status (dict): A dictionary tracking if the pellet in a given discrete state has been collected.
        bonus (float): The bonus to add if the state is new.
        penalty (float): The penalty to add if the state has been visited before.

    Returns:
        float: The shaped reward.
    """
    if discrete_state not in pellet_status:
        pellet_status[discrete_state] = True  # Mark as visited (pellet collected)
        return original_reward + bonus
    else:
        return original_reward + penalty