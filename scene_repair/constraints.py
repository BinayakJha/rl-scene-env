"""The three kinds of goals ("constraints") and how to check them.

    Center(k)          object k is inside the middle 3x3 square: x and y are both in {1, 2, 3}
    Adjacent(i, j)     objects i and j touch side-by-side: Manhattan distance exactly 1
    Direction(i, j, d) object i is exactly ONE cell in direction d from object j
                       (example: Direction(book, box, "S") means book is directly south of box)

The checker here is exact. Every policy and the reward use these same functions.
"""
from dataclasses import dataclass

from .grid import DIRECTIONS, manhattan, step


@dataclass(frozen=True)
class Constraint:
    kind: str            # "center", "adjacent" or "direction"
    i: int               # the main object
    j: int = -1          # the second object (not used by "center")
    direction: str = ""  # only used by "direction"

    def describe(self, names):
        """Plain-English text, e.g. 'book is directly S of box'."""
        if self.kind == "center":
            return f"{names[self.i]} is in the center 3x3"
        if self.kind == "adjacent":
            return f"{names[self.i]} is next to {names[self.j]}"
        return f"{names[self.i]} is directly {self.direction} of {names[self.j]}"


def is_satisfied(constraint, layout):
    """True if this one constraint holds in this layout."""
    a = layout[constraint.i]
    if constraint.kind == "center":
        return 1 <= a[0] <= 3 and 1 <= a[1] <= 3
    b = layout[constraint.j]
    if constraint.kind == "adjacent":
        return manhattan(a, b) == 1
    if constraint.kind == "direction":
        return a == step(b, constraint.direction)
    raise ValueError(f"unknown constraint kind {constraint.kind}")


def satisfaction_mask(constraints, layout):
    """The mask m: one True/False per constraint, in order."""
    return tuple(is_satisfied(c, layout) for c in constraints)


def fraction_satisfied(constraints, layout):
    """s in the reward formula: the share of constraints that hold (0.0 to 1.0)."""
    mask = satisfaction_mask(constraints, layout)
    return sum(mask) / len(mask)


def moves_needed_at_least(constraint, layout):
    """A lower bound on how many moves this constraint still needs.

    Each move shifts ONE object by ONE cell, so it can shrink each gap below by at most 1.
    The planner uses this as a safe (never too big) A* estimate.
    """
    a = layout[constraint.i]
    if constraint.kind == "center":
        dx = max(0, 1 - a[0], a[0] - 3)
        dy = max(0, 1 - a[1], a[1] - 3)
        return dx + dy
    b = layout[constraint.j]
    if constraint.kind == "adjacent":
        return max(0, manhattan(a, b) - 1)
    # direction: object i must land on the one cell next to j in that direction
    dx, dy = DIRECTIONS[constraint.direction]
    return abs(a[0] - (b[0] + dx)) + abs(a[1] - (b[1] + dy))
