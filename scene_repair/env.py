"""The Gymnasium environment: the exact rules of Budgeted Stochastic Scene Repair.

State (all visible to the player):  C constraints, L layout, b budget, m satisfied-mask,
                                    h undo buffer (one saved layout or empty), o last outcome.
Actions (always 22 numbers, because we pad to 5 objects):
    action = 4*k + d   MOVE object k in direction d (d: 0=N, 1=S, 2=E, 3=W)   -> actions 0..19
    action = 20        UNDO
    action = 21        STOP
Rules:
  * MOVE and UNDO cost 1 budget, even when nothing changes. STOP is free and ends the episode.
  * When the budget hits 0 the episode also ends.
  * A MOVE onto an occupied cell or off the board is "invalid": nothing moves, still charged.
  * A legal MOVE: 70% lands where intended, 20% "jitters" to a random empty neighbor (8-way)
    of the intended cell, 10% fails (nothing moves).
  * Every MOVE saves the layout from before it into h. UNDO puts that layout back and empties h.
    UNDO with an empty h does nothing but is still charged. Budget is never refunded.
  * Reward is 0 on every step except the last one, where R = s * (0.75 + 0.25 * b_end / 8).

Padding choice: scenes have 3-5 objects, but the action list always has 22 entries.
Actions for objects that do not exist are switched off in info["action_mask"]. If a player
picks one anyway, it is treated like an invalid move (charged, nothing happens).
Out-of-bounds and occupied-target moves are NOT masked: they are real (bad) choices in the MDP.
"""
import gymnasium as gym
import numpy as np

from .constraints import fraction_satisfied, satisfaction_mask
from .encoding import OBS_SIZE, encode
from .grid import DIRECTION_NAMES, in_bounds, moore_neighbors, move_object, step
from .scenes import MAX_OBJECTS, scene_for_split

NUM_ACTIONS = 4 * MAX_OBJECTS + 2  # 22
UNDO = 4 * MAX_OBJECTS             # 20
STOP = UNDO + 1                    # 21
OUTCOMES = ["start", "success", "jitter", "fail", "invalid", "undo", "undo_empty", "stop"]


def action_name(action, names=None):
    """Human-readable action, e.g. 'MOVE(box, E)'."""
    if action == UNDO:
        return "UNDO"
    if action == STOP:
        return "STOP"
    k, d = divmod(action, 4)
    label = names[k] if names and k < len(names) else f"obj{k}"
    return f"MOVE({label}, {DIRECTION_NAMES[d]})"


def terminal_reward(s, budget_left, start_budget):
    """R = s * (0.75 + 0.25 * b_end / B0). Full success with budget left scores highest."""
    return s * (0.75 + 0.25 * budget_left / start_budget)


class SceneRepairEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, split="train", budget=8, noise=(0.7, 0.2, 0.1), allow_undo=True):
        """
        split:      which scenes reset() draws when no scene is given ("train", "val", "test")
        budget:     B0, the starting budget (8 in the proposal)
        noise:      (success, jitter, fail) chances for a legal move. (1, 0, 0) = no noise.
        allow_undo: False switches UNDO off (used only for the "no UNDO" ablation)
        """
        self.split = split
        self.start_budget = budget
        self.noise = noise
        self.allow_undo = allow_undo
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        """Start a new episode. Pass options={"scene": scene} to play a specific scene."""
        super().reset(seed=seed)  # seeds self.np_random, used for the move noise
        if options and "scene" in options:
            self.scene = options["scene"]
        else:
            scene_seed = seed if seed is not None else int(self.np_random.integers(2**31))
            self.scene = scene_for_split(self.split, scene_seed)
        self.constraints = self.scene.constraints
        self.layout = self.scene.start_layout
        self.budget = self.start_budget
        self.undo_buffer = None   # h starts empty
        self.outcome = "start"    # o
        self.done = False
        return self._obs(), self._info()

    # ------------------------------------------------------------------- step
    def step(self, action):
        assert not self.done, "episode is over; call reset()"
        action = int(action)

        if action == STOP:
            self.outcome = "stop"
            self.done = True  # STOP is free: budget is not touched
        else:
            self.budget -= 1  # every MOVE and UNDO costs one unit, whatever happens
            if action == UNDO:
                self._do_undo()
            else:
                self._do_move(action)
            if self.budget == 0:
                self.done = True  # out of budget: the episode is forced to end

        reward = 0.0
        if self.done:
            s = fraction_satisfied(self.constraints, self.layout)
            reward = terminal_reward(s, self.budget, self.start_budget)
        # terminated=True means the episode really ended; we never use truncation
        return self._obs(), reward, self.done, False, self._info()

    def _do_undo(self):
        if not self.allow_undo or self.undo_buffer is None:
            self.outcome = "undo_empty"  # charged no-op
            return
        self.layout = self.undo_buffer  # deterministic restore
        self.undo_buffer = None         # buffer is cleared after use
        self.outcome = "undo"

    def _do_move(self, action):
        k, d = divmod(action, 4)
        before = self.layout
        self.undo_buffer = before  # every MOVE saves the pre-move layout (length-one buffer)

        if k >= len(self.layout):
            self.outcome = "invalid"  # padded slot: this object does not exist
            return
        u = before[k]                        # where the object is now
        v = step(u, DIRECTION_NAMES[d])      # where we want it to go
        if not in_bounds(v) or v in before:
            self.outcome = "invalid"         # off the board or onto another object
            return

        p_success, p_jitter, _ = self.noise
        roll = self.np_random.random()
        if roll < p_success:
            self.layout = move_object(before, k, v)
            self.outcome = "success"
        elif roll < p_success + p_jitter:
            self.outcome = "jitter"
            candidates = jitter_cells(before, u, v)
            if candidates:  # pick one at random; if none, nothing moves
                pick = candidates[self.np_random.integers(len(candidates))]
                self.layout = move_object(before, k, pick)
        else:
            self.outcome = "fail"

    # ------------------------------------------------------------ what we show
    def state(self):
        """The full readable state (C, L, b, m, h, o). The planner reads this directly."""
        return {
            "constraints": self.constraints,
            "layout": self.layout,
            "budget": self.budget,
            "mask": satisfaction_mask(self.constraints, self.layout),
            "undo_buffer": self.undo_buffer,
            "outcome": self.outcome,
            "names": self.scene.names,
            "start_budget": self.start_budget,
        }

    def action_mask(self):
        """True for actions that refer to a real object (plus UNDO if allowed, plus STOP)."""
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        mask[: 4 * len(self.layout)] = True
        mask[UNDO] = self.allow_undo
        mask[STOP] = True
        return mask

    def _obs(self):
        return encode(self.state())

    def _info(self):
        return {"state": self.state(), "action_mask": self.action_mask()}


def jitter_cells(layout_before, u, v):
    """Where a jittered move can land: empty, on-board 8-way neighbors of the target v.

    "Empty" is judged on the grid BEFORE the move. The origin u is occupied then, so it is
    automatically left out (we also remove it explicitly to make the rule obvious).
    """
    occupied = set(layout_before)
    return [c for c in moore_neighbors(v) if c not in occupied and c != u]
