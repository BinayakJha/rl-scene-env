"""The environment rules: collisions, noise, jitter, undo, budget, STOP and reward."""
import pytest

from helpers import FAIL, JITTER, SUCCESS, make_env_with, worked_example_scene
from scene_repair.constraints import Constraint
from scene_repair.env import NUM_ACTIONS, STOP, UNDO, SceneRepairEnv, jitter_cells
from scene_repair.scenes import Scene

N, S, E, W = 0, 1, 2, 3


def move(k, d):
    return 4 * k + d


def two_object_scene(a=(2, 2), b=(4, 4)):
    return Scene((Constraint("adjacent", 0, 1), Constraint("center", 0)), (a, b), (a, (2, 3)))


def test_worked_example_from_the_proposal_gives_0_875():
    env = make_env_with(worked_example_scene(), rolls=[FAIL, SUCCESS, SUCCESS, SUCCESS])
    box, cat, book = 0, 1, 2
    expected = [
        (move(box, E), "fail", ((0, 1), (3, 1), (1, 3)), 7, (False, False, False)),
        (move(box, E), "success", ((1, 1), (3, 1), (1, 3)), 6, (True, False, False)),
        (move(cat, W), "success", ((1, 1), (2, 1), (1, 3)), 5, (True, True, False)),
        (move(book, N), "success", ((1, 1), (2, 1), (1, 2)), 4, (True, True, True)),
    ]
    for action, outcome, layout, budget, mask in expected:
        _, reward, done, truncated, info = env.step(action)
        state = info["state"]
        assert (state["outcome"], state["layout"], state["budget"], state["mask"]) == \
               (outcome, layout, budget, mask)
        assert reward == 0.0 and not done and not truncated  # no intermediate reward
    _, reward, done, _, info = env.step(STOP)
    assert done and info["state"]["budget"] == 4       # STOP is free
    assert reward == pytest.approx(0.875)              # 1 * (0.75 + 0.25 * 4/8)


def test_out_of_bounds_move_is_charged_and_changes_nothing():
    env = make_env_with(two_object_scene(a=(0, 0)))
    _, _, _, _, info = env.step(move(0, N))
    assert info["state"]["outcome"] == "invalid"
    assert info["state"]["layout"] == ((0, 0), (4, 4))
    assert info["state"]["budget"] == 7


def test_move_onto_occupied_cell_is_charged_and_changes_nothing():
    env = make_env_with(two_object_scene(a=(2, 2), b=(3, 2)))
    _, _, _, _, info = env.step(move(0, E))
    assert info["state"]["outcome"] == "invalid"
    assert info["state"]["layout"] == ((2, 2), (3, 2))
    assert info["state"]["budget"] == 7


def test_failed_move_changes_nothing_but_costs_one():
    env = make_env_with(two_object_scene(), rolls=[FAIL])
    _, _, _, _, info = env.step(move(0, E))
    assert info["state"]["outcome"] == "fail"
    assert info["state"]["layout"] == ((2, 2), (4, 4))
    assert info["state"]["budget"] == 7


def test_jitter_candidates_are_empty_moore_neighbors_of_target_without_origin():
    # object at u=(2,2) wants v=(3,2); another object sits at (4,1)
    cells = jitter_cells(((2, 2), (4, 1)), u=(2, 2), v=(3, 2))
    assert set(cells) == {(2, 1), (3, 1), (4, 2), (2, 3), (3, 3), (4, 3)}
    assert (2, 2) not in cells and (4, 1) not in cells and (3, 2) not in cells


def test_jitter_candidates_stay_on_the_board():
    cells = jitter_cells(((1, 0),), u=(1, 0), v=(0, 0))
    assert set(cells) == {(0, 1), (1, 1)}


def test_jitter_moves_to_a_candidate():
    env = make_env_with(two_object_scene(), rolls=[JITTER], picks=[0])
    _, _, _, _, info = env.step(move(0, E))
    candidates = jitter_cells(((2, 2), (4, 4)), (2, 2), (3, 2))
    assert info["state"]["outcome"] == "jitter"
    assert info["state"]["layout"][0] == candidates[0]


def test_jitter_with_no_candidates_leaves_layout_unchanged():
    # box at (1,0) moves W to v=(0,0). v's neighbors: (1,0)=origin, (0,1) and (1,1) are full.
    scene = Scene((Constraint("center", 0), Constraint("adjacent", 1, 2)),
                  ((1, 0), (0, 1), (1, 1)), ((1, 1), (0, 1), (0, 2)))
    env = make_env_with(scene, rolls=[JITTER])
    _, _, _, _, info = env.step(move(0, W))
    assert info["state"]["outcome"] == "jitter"
    assert info["state"]["layout"] == ((1, 0), (0, 1), (1, 1))
    assert info["state"]["budget"] == 7


def test_undo_restores_previous_layout_clears_buffer_and_costs_one():
    env = make_env_with(two_object_scene(), rolls=[SUCCESS])
    _, _, _, _, info = env.step(move(0, E))
    assert info["state"]["undo_buffer"] == ((2, 2), (4, 4))
    _, _, _, _, info = env.step(UNDO)
    assert info["state"]["outcome"] == "undo"
    assert info["state"]["layout"] == ((2, 2), (4, 4))
    assert info["state"]["undo_buffer"] is None
    assert info["state"]["budget"] == 6  # no refund


def test_undo_with_empty_buffer_is_a_charged_no_op():
    env = make_env_with(two_object_scene())
    _, _, _, _, info = env.step(UNDO)
    assert info["state"]["outcome"] == "undo_empty"
    assert info["state"]["layout"] == ((2, 2), (4, 4))
    assert info["state"]["budget"] == 7


def test_undo_buffer_has_length_one():
    env = make_env_with(two_object_scene(), rolls=[SUCCESS, SUCCESS])
    env.step(move(0, E))   # (2,2) -> (3,2)
    env.step(move(0, S))   # (3,2) -> (3,3)
    _, _, _, _, info = env.step(UNDO)
    assert info["state"]["layout"] == ((3, 2), (4, 4))  # only the LAST move is undone
    _, _, _, _, info = env.step(UNDO)
    assert info["state"]["outcome"] == "undo_empty"


def test_stop_is_free_and_ends_episode():
    env = make_env_with(two_object_scene())
    _, reward, done, _, info = env.step(STOP)
    assert done and info["state"]["budget"] == 8
    # mask is (adjacent False, center True) -> s = 0.5, full budget left
    assert reward == pytest.approx(0.5 * 1.0)


def test_budget_zero_forces_the_end():
    env = make_env_with(two_object_scene(), rolls=[FAIL] * 8)
    for t in range(8):
        _, reward, done, _, info = env.step(move(0, E))
        assert done == (t == 7)
    assert info["state"]["budget"] == 0
    assert reward == pytest.approx(0.5 * 0.75)  # s = 0.5, no budget left
    with pytest.raises(AssertionError):
        env.step(STOP)


def test_padded_action_for_missing_object_is_masked_and_charged():
    env = make_env_with(two_object_scene())
    mask = env.action_mask()
    assert len(mask) == NUM_ACTIONS == 22
    assert mask[:8].all() and not mask[8:20].any() and mask[UNDO] and mask[STOP]
    _, _, _, _, info = env.step(move(4, N))
    assert info["state"]["outcome"] == "invalid" and info["state"]["budget"] == 7


def test_no_undo_ablation_switches_undo_off():
    env = make_env_with(two_object_scene(), rolls=[SUCCESS], allow_undo=False)
    assert not env.action_mask()[UNDO]
    env.step(move(0, E))
    _, _, _, _, info = env.step(UNDO)
    assert info["state"]["outcome"] == "undo_empty"
    assert info["state"]["layout"] == ((3, 2), (4, 4))


def test_noise_rates_are_roughly_right():
    env = SceneRepairEnv()
    counts = {"success": 0, "jitter": 0, "fail": 0}
    scene = two_object_scene()
    for seed in range(3000):
        env.reset(seed=seed, options={"scene": scene})
        _, _, _, _, info = env.step(move(0, E))
        counts[info["state"]["outcome"]] += 1
    assert abs(counts["success"] / 3000 - 0.7) < 0.03
    assert abs(counts["jitter"] / 3000 - 0.2) < 0.03
    assert abs(counts["fail"] / 3000 - 0.1) < 0.03
