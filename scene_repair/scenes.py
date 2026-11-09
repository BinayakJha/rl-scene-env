"""Making puzzles ("scenes") and splitting them into train / validation / test.

How one scene is made (the proposal's recipe):
  1. Pick K = 3..5 objects and drop them on random different cells. This is the "witness":
     a layout we KNOW is a solution.
  2. Pick 2..4 constraints that are TRUE in the witness. So every puzzle is solvable.
  3. Scramble the witness with 1..6 random legal moves (no noise). Playing those moves
     backwards solves the puzzle, so a noise-free solution always fits in the budget of 8.
  4. Throw the scene away if it is already solved after scrambling.

How splits stay apart:
  * Each split uses its own range of random seeds.
  * A scene's "template" is its exact goal list (which goal types, which objects, which
    directions). We hash the template and the hash decides the split. So a goal list that
    appears in the test set NEVER appears in training.
"""
import hashlib
import json
from dataclasses import dataclass

import numpy as np

from .constraints import Constraint, satisfaction_mask
from .grid import DIRECTION_NAMES, all_cells, in_bounds, manhattan, move_object, step

OBJECT_NAMES = ["box", "cat", "book", "lamp", "cup"]  # object k is always called OBJECT_NAMES[k]
MAX_OBJECTS = 5
MAX_CONSTRAINTS = 4
MAX_SCRAMBLE_MOVES = 6

# Where each split's seeds start. The gaps are huge, so the ranges never overlap.
SPLIT_SEED_START = {"train": 0, "val": 1_000_000_000, "test": 2_000_000_000}


@dataclass(frozen=True)
class Scene:
    """One puzzle: the goals, the starting layout, and (for checking) a known solution."""
    constraints: tuple   # tuple of Constraint
    start_layout: tuple  # tuple of (x, y), one per object
    witness_layout: tuple  # a layout where every constraint holds

    @property
    def num_objects(self):
        return len(self.start_layout)

    @property
    def names(self):
        return OBJECT_NAMES[: self.num_objects]


def template_key(constraints):
    """A text fingerprint of the goal list, e.g. 'center(0)|adjacent(1,0)|direction(2,0,S)'."""
    parts = []
    for c in constraints:
        if c.kind == "center":
            parts.append(f"center({c.i})")
        elif c.kind == "adjacent":
            parts.append(f"adjacent({min(c.i, c.j)},{max(c.i, c.j)})")  # order does not matter
        else:
            parts.append(f"direction({c.i},{c.j},{c.direction})")
    return "|".join(parts)


def split_of_template(key):
    """Hash the template text into a bucket 0-9: 0-6 train, 7 val, 8-9 test."""
    bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10
    if bucket <= 6:
        return "train"
    if bucket == 7:
        return "val"
    return "test"


def true_constraints_in(layout):
    """Every constraint that holds in this layout (the menu we pick goals from)."""
    options = []
    k_count = len(layout)
    for i in range(k_count):
        if 1 <= layout[i][0] <= 3 and 1 <= layout[i][1] <= 3:
            options.append(Constraint("center", i))
    for i in range(k_count):
        for j in range(k_count):
            if i == j:
                continue
            for d in DIRECTION_NAMES:
                if layout[i] == step(layout[j], d):
                    options.append(Constraint("direction", i, j, d))
            if i < j and manhattan(layout[i], layout[j]) == 1:
                options.append(Constraint("adjacent", i, j))
    return options


def legal_moves(layout):
    """All (object, direction) moves whose target is on the board and empty."""
    occupied = set(layout)
    moves = []
    for k, cell in enumerate(layout):
        for d in DIRECTION_NAMES:
            target = step(cell, d)
            if in_bounds(target) and target not in occupied:
                moves.append((k, d))
    return moves


def _pick_constraints(rng, layout, how_many):
    """Pick `how_many` true constraints, at least one between two objects, no repeated pair."""
    options = true_constraints_in(layout)
    relations = [c for c in options if c.kind != "center"]
    if not relations:
        return None  # this witness has no touching objects; caller will try again
    chosen = [relations[rng.integers(len(relations))]]
    used_pairs = {frozenset((chosen[0].i, chosen[0].j))}
    used_centers = set()
    for index in rng.permutation(len(options)):
        if len(chosen) == how_many:
            break
        c = options[index]
        if c.kind == "center":
            if c.i in used_centers:
                continue
            used_centers.add(c.i)
        else:
            pair = frozenset((c.i, c.j))
            if pair in used_pairs:
                continue  # e.g. skip "adjacent(box,cat)" if we already have "cat S of box"
            used_pairs.add(pair)
        chosen.append(c)
    if len(chosen) < how_many:
        return None
    order = rng.permutation(len(chosen))
    return tuple(chosen[i] for i in order)


def generate_scene(rng):
    """Make one random scene with the recipe at the top of this file."""
    while True:
        k_count = int(rng.integers(3, MAX_OBJECTS + 1))
        cells = all_cells()
        picks = rng.choice(len(cells), size=k_count, replace=False)
        witness = tuple(cells[p] for p in picks)
        how_many = int(rng.integers(2, MAX_CONSTRAINTS + 1))
        constraints = _pick_constraints(rng, witness, how_many)
        if constraints is None:
            continue
        layout = witness
        for _ in range(int(rng.integers(1, MAX_SCRAMBLE_MOVES + 1))):
            k, d = legal_moves(layout)[rng.integers(len(legal_moves(layout)))]
            layout = move_object(layout, k, step(layout[k], d))
        if all(satisfaction_mask(constraints, layout)):
            continue  # already solved: reject, as the proposal says
        return Scene(constraints, layout, witness)


def scene_for_split(split, seed):
    """The scene for a given split and seed. Keeps drawing until the template belongs to `split`."""
    rng = np.random.default_rng(SPLIT_SEED_START[split] + seed)
    while True:
        scene = generate_scene(rng)
        if split_of_template(template_key(scene.constraints)) == split:
            return scene


def make_fixed_set(split, count):
    """A frozen list of `count` different scenes (no repeats) for validation or testing."""
    scenes, seen, seed = [], set(), 0
    while len(scenes) < count:
        scene = scene_for_split(split, seed)
        seed += 1
        fingerprint = (template_key(scene.constraints), scene.start_layout)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        scenes.append(scene)
    return scenes


# ---- saving and loading frozen sets as JSON (so they are easy to inspect by eye) ----

def scene_to_dict(scene):
    return {
        "constraints": [[c.kind, c.i, c.j, c.direction] for c in scene.constraints],
        "start_layout": [list(p) for p in scene.start_layout],
        "witness_layout": [list(p) for p in scene.witness_layout],
        "template": template_key(scene.constraints),
    }


def scene_from_dict(data):
    return Scene(
        constraints=tuple(Constraint(kind, i, j, d) for kind, i, j, d in data["constraints"]),
        start_layout=tuple(tuple(p) for p in data["start_layout"]),
        witness_layout=tuple(tuple(p) for p in data["witness_layout"]),
    )


def save_scenes(scenes, path):
    with open(path, "w") as f:
        json.dump([scene_to_dict(s) for s in scenes], f, indent=1)


def load_scenes(path):
    with open(path) as f:
        return [scene_from_dict(d) for d in json.load(f)]
