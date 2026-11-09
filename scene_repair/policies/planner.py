"""Adaptive replanner: the strong non-learning baseline.

Idea in one line: pretend moves never fail, find the shortest fix, do its FIRST step,
look at what really happened, and plan again from there.

Details:
  * Search method: A* search over layouts in the noise-free ("deterministic relaxation") world.
    A* = always expand the layout with the smallest (moves so far + estimate of moves left).
  * Estimate of moves left: the largest per-goal lower bound (constraints.moves_needed_at_least).
    It never over-estimates, so the first full solution A* finds is a shortest one.
  * UNDO is also an option at the first step, because it restores the saved layout with no noise.
  * Search never goes deeper than the budget left, and stops after MAX_EXPANSIONS layouts.
  * If no full fix fits, it heads for the layout with the best noise-free reward it saw.
  * If the scene is solved (or nothing beats stopping now), it says STOP.
"""
import heapq
import itertools

from ..constraints import fraction_satisfied, moves_needed_at_least, satisfaction_mask
from ..env import STOP, UNDO, terminal_reward
from ..grid import DIRECTION_NAMES, in_bounds, move_object, step

MAX_EXPANSIONS = 20_000


def lower_bound(constraints, layout):
    return max(moves_needed_at_least(c, layout) for c in constraints)


def successors(layout):
    """Every legal noise-free move from this layout, as (action number, new layout)."""
    occupied = set(layout)
    for k, cell in enumerate(layout):
        for d, name in enumerate(DIRECTION_NAMES):
            target = step(cell, name)
            if in_bounds(target) and target not in occupied:
                yield 4 * k + d, move_object(layout, k, target)


def plan(state, max_expansions=MAX_EXPANSIONS, allow_undo=True):
    """Return (list of actions, number of layouts expanded). An empty list means: STOP now."""
    constraints, start, budget = state["constraints"], state["layout"], state["budget"]
    b0 = state["start_budget"]

    def score(layout, used):  # the noise-free reward if we stopped at this layout
        return terminal_reward(fraction_satisfied(constraints, layout), budget - used, b0)

    came_from = {start: None}   # layout -> (previous layout, action)
    cost = {start: 0}
    best_layout, best_score = start, score(start, 0)
    tie = itertools.count()     # breaks ties in the heap without comparing layouts
    heap = [(lower_bound(constraints, start), 0, next(tie), start)]
    expansions = 0

    while heap and expansions < max_expansions:
        _, g, _, layout = heapq.heappop(heap)
        if g > cost[layout]:
            continue  # stale heap entry; a shorter way here was already found
        if all(satisfaction_mask(constraints, layout)):
            best_layout = layout  # A* pops goals in order of cost, so this one is shortest
            break
        expansions += 1
        if g == budget:
            continue  # no budget left to go deeper
        options = list(successors(layout))
        if g == 0 and allow_undo and state["undo_buffer"] is not None:
            options.append((UNDO, state["undo_buffer"]))
        for action, nxt in options:
            if nxt not in cost or g + 1 < cost[nxt]:
                cost[nxt] = g + 1
                came_from[nxt] = (layout, action)
                s = score(nxt, g + 1)
                if s > best_score:
                    best_layout, best_score = nxt, s
                heapq.heappush(heap, (g + 1 + lower_bound(constraints, nxt), g + 1, next(tie), nxt))

    # walk back from the chosen layout to the start to read off the actions
    actions = []
    layout = best_layout
    while came_from[layout] is not None:
        layout, action = came_from[layout]
        actions.append(action)
    return actions[::-1], expansions


class PlannerPolicy:
    name = "planner"

    def __init__(self, max_expansions=MAX_EXPANSIONS, allow_undo=True):
        self.max_expansions = max_expansions
        self.allow_undo = allow_undo
        self.total_expansions = 0  # for reporting search compute
        self.limit_hits = 0        # decisions where the search ran out of expansions

    def act(self, obs, info):
        state = info["state"]
        if all(state["mask"]):
            return STOP  # solved: stop and keep the leftover budget
        actions, expansions = plan(state, self.max_expansions, self.allow_undo)
        self.total_expansions += expansions
        self.limit_hits += expansions >= self.max_expansions
        return actions[0] if actions else STOP
