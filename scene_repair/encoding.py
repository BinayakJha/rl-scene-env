"""Turn the readable state (C, L, b, m, h, o) into one flat list of numbers for a neural net.

Neural nets need fixed-size number lists, so every part is padded to the maximum size
(5 objects, 4 constraints). The vector has these blocks, in order:

  objects      5 x [present, x one-hot(5), y one-hot(5), can-move N/S/E/W(4),
                    pull dx, pull dy]                                         = 85
  constraints  4 x [present, kind one-hot(3), i one-hot(5), j one-hot(5),
                    direction one-hot(4), satisfied, gap dx, gap dy]           = 84
  budget       one-hot of b (0..8) and b / B0                                  = 10
  undo buffer  [has a saved layout] + saved layout as 5 x [present, x(5), y(5)] = 56
  last outcome one-hot over the 8 outcome names                                = 8
                                                                         total = 243

"can-move", "pull" and "gap" are HINTS: they are computed only from C and L, so they add no
new information. They just save the small network from having to learn arithmetic.
  gap  = for one goal: how far (in x and in y) object i still is from where that goal wants it.
  pull = for one object: the gaps of all its UNMET goals added up ("which way it wants to go").
         If the object is the second one (j) in a goal, the gap counts with a minus sign,
         because moving j the other way closes the same gap.
All of these are divided by 4 so they sit roughly between -1 and 1.
"""
import numpy as np

from .grid import DIRECTION_NAMES, DIRECTIONS, GRID_SIZE, in_bounds, step

MAX_OBJECTS = 5
MAX_CONSTRAINTS = 4
MAX_BUDGET = 8
KINDS = ["center", "adjacent", "direction"]
OUTCOME_NAMES = ["start", "success", "jitter", "fail", "invalid", "undo", "undo_empty", "stop"]

OBJECT_BLOCK = 1 + GRID_SIZE + GRID_SIZE + 4 + 2
CONSTRAINT_BLOCK = 1 + 3 + MAX_OBJECTS + MAX_OBJECTS + 4 + 1 + 2
LAYOUT_BLOCK = 1 + GRID_SIZE + GRID_SIZE
OBS_SIZE = (MAX_OBJECTS * OBJECT_BLOCK + MAX_CONSTRAINTS * CONSTRAINT_BLOCK
            + (MAX_BUDGET + 2) + (1 + MAX_OBJECTS * LAYOUT_BLOCK) + len(OUTCOME_NAMES))


def one_hot(index, size):
    v = np.zeros(size, dtype=np.float32)
    v[index] = 1.0
    return v


def encode_layout(layout):
    """5 x [present, x one-hot, y one-hot]; missing objects are all zeros."""
    parts = []
    for k in range(MAX_OBJECTS):
        if k < len(layout):
            x, y = layout[k]
            parts += [[1.0], one_hot(x, GRID_SIZE), one_hot(y, GRID_SIZE)]
        else:
            parts.append(np.zeros(LAYOUT_BLOCK, dtype=np.float32))
    return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])


def shrink_by_one(dx, dy):
    """For "adjacent": object i only needs to get within 1 of j, so drop one step of the gap."""
    if abs(dx) >= abs(dy):
        return dx - (1 if dx > 0 else -1 if dx < 0 else 0), dy
    return dx, dy - (1 if dy > 0 else -1)


def goal_gap(constraint, layout):
    """(dx, dy) object i still has to travel to meet this goal, divided by 4."""
    ax, ay = layout[constraint.i]
    if constraint.kind == "center":
        dx = 1 - ax if ax < 1 else (3 - ax if ax > 3 else 0)
        dy = 1 - ay if ay < 1 else (3 - ay if ay > 3 else 0)
    elif constraint.kind == "adjacent":
        bx, by = layout[constraint.j]
        dx, dy = shrink_by_one(bx - ax, by - ay)  # travel to a cell next to j
    else:
        bx, by = layout[constraint.j]
        ddx, ddy = DIRECTIONS[constraint.direction]
        dx, dy = (bx + ddx) - ax, (by + ddy) - ay
    return np.array([dx / 4.0, dy / 4.0], dtype=np.float32)


def object_pulls(state):
    """Sum of the gaps of each object's unmet goals (see the top of this file)."""
    pulls = np.zeros((MAX_OBJECTS, 2), dtype=np.float32)
    for c, ok in zip(state["constraints"], state["mask"]):
        if ok:
            continue
        gap = goal_gap(c, state["layout"])
        pulls[c.i] += gap
        if c.j >= 0:
            pulls[c.j] -= gap
    return pulls


def encode(state):
    layout = state["layout"]
    occupied = set(layout)
    pulls = object_pulls(state)
    parts = []

    # objects: where each one is, and which of its 4 moves would be legal right now
    for k in range(MAX_OBJECTS):
        if k < len(layout):
            x, y = layout[k]
            free = [float(in_bounds(t) and t not in occupied)
                    for t in (step(layout[k], d) for d in DIRECTION_NAMES)]
            parts += [[1.0], one_hot(x, GRID_SIZE), one_hot(y, GRID_SIZE), free, pulls[k]]
        else:
            parts.append(np.zeros(OBJECT_BLOCK, dtype=np.float32))

    # constraints: what each goal says, whether it holds, and how far off it is
    for n in range(MAX_CONSTRAINTS):
        if n < len(state["constraints"]):
            c = state["constraints"][n]
            j_part = one_hot(c.j, MAX_OBJECTS) if c.j >= 0 else np.zeros(MAX_OBJECTS)
            d_part = (one_hot(DIRECTION_NAMES.index(c.direction), 4)
                      if c.direction else np.zeros(4))
            parts += [[1.0], one_hot(KINDS.index(c.kind), 3), one_hot(c.i, MAX_OBJECTS),
                      j_part, d_part, [float(state["mask"][n])], goal_gap(c, layout)]
        else:
            parts.append(np.zeros(CONSTRAINT_BLOCK, dtype=np.float32))

    # budget
    b = state["budget"]
    parts += [one_hot(min(b, MAX_BUDGET), MAX_BUDGET + 1), [b / state["start_budget"]]]

    # undo buffer
    h = state["undo_buffer"]
    if h is None:
        parts.append(np.zeros(1 + MAX_OBJECTS * LAYOUT_BLOCK, dtype=np.float32))
    else:
        parts += [[1.0], encode_layout(h)]

    # last outcome
    parts.append(one_hot(OUTCOME_NAMES.index(state["outcome"]), len(OUTCOME_NAMES)))

    vector = np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])
    assert vector.shape == (OBS_SIZE,)
    return vector
