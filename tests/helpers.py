"""Test helpers: a fake random-number source so tests can force success, jitter or failure."""
from scene_repair.constraints import Constraint
from scene_repair.scenes import Scene

# Rolls that force each outcome with noise (0.7, 0.2, 0.1):
SUCCESS, JITTER, FAIL = 0.0, 0.75, 0.95


class FakeRandom:
    """Stands in for env.np_random. random() returns the next queued roll;
    integers(n) returns the next queued index (default 0)."""

    def __init__(self, rolls=(), picks=()):
        self.rolls, self.picks = list(rolls), list(picks)

    def random(self):
        return self.rolls.pop(0)

    def integers(self, n):
        return self.picks.pop(0) if self.picks else 0


def worked_example_scene():
    """The proposal's example: box (0,1), cat (3,1), book (1,3) and three goals."""
    constraints = (
        Constraint("center", 0),               # box is centered
        Constraint("adjacent", 1, 0),          # cat is next to box
        Constraint("direction", 2, 0, "S"),    # book is directly south of box
    )
    return Scene(constraints, ((0, 1), (3, 1), (1, 3)), ((1, 1), (2, 1), (1, 2)))


def make_env_with(scene, rolls=(), picks=(), **kwargs):
    from scene_repair.env import SceneRepairEnv
    env = SceneRepairEnv(**kwargs)
    env.reset(seed=0, options={"scene": scene})
    env.np_random = FakeRandom(rolls, picks)
    return env
