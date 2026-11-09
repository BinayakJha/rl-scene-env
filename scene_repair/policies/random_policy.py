"""Random policy: picks any allowed action uniformly at random. It is the "floor" to beat."""
import numpy as np


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, info):
        allowed = np.flatnonzero(info["action_mask"])  # real objects, UNDO, STOP
        return int(self.rng.choice(allowed))
