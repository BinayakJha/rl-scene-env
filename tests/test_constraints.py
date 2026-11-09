"""Predicates use (x, y) with y growing SOUTH, exactly as the proposal says."""
from scene_repair.constraints import Constraint, is_satisfied, moves_needed_at_least


def test_center_is_the_middle_3x3():
    c = Constraint("center", 0)
    for x in range(5):
        for y in range(5):
            expected = x in (1, 2, 3) and y in (1, 2, 3)
            assert is_satisfied(c, ((x, y),)) == expected


def test_adjacent_means_manhattan_distance_one():
    c = Constraint("adjacent", 0, 1)
    assert is_satisfied(c, ((2, 2), (2, 3)))
    assert is_satisfied(c, ((2, 2), (1, 2)))
    assert not is_satisfied(c, ((2, 2), (3, 3)))  # diagonal is NOT adjacent
    assert not is_satisfied(c, ((2, 2), (2, 4)))


def test_direction_south_means_y_plus_one():
    south = Constraint("direction", 0, 1, "S")  # object 0 directly south of object 1
    assert is_satisfied(south, ((1, 2), (1, 1)))
    assert not is_satisfied(south, ((1, 0), (1, 1)))  # that is north
    north = Constraint("direction", 0, 1, "N")
    assert is_satisfied(north, ((1, 0), (1, 1)))
    east = Constraint("direction", 0, 1, "E")
    assert is_satisfied(east, ((2, 1), (1, 1)))


def test_lower_bound_is_zero_when_satisfied_and_never_too_big():
    assert moves_needed_at_least(Constraint("center", 0), ((2, 2),)) == 0
    assert moves_needed_at_least(Constraint("center", 0), ((0, 0),)) == 2
    assert moves_needed_at_least(Constraint("adjacent", 0, 1), ((0, 0), (4, 0))) == 3
    assert moves_needed_at_least(Constraint("direction", 0, 1, "S"), ((1, 3), (1, 1))) == 1
