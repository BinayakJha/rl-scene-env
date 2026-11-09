"""Reset generator: solvable, not already solved, within reach, and splits never overlap."""
from scene_repair.constraints import satisfaction_mask
from scene_repair.policies.planner import plan
from scene_repair.scenes import (make_fixed_set, scene_for_split, split_of_template,
                                 template_key)


def test_scenes_are_valid_and_unsolved():
    for seed in range(300):
        scene = scene_for_split("train", seed)
        assert 3 <= scene.num_objects <= 5
        assert 2 <= len(scene.constraints) <= 4
        assert len(set(scene.start_layout)) == scene.num_objects   # no two objects share a cell
        assert all(satisfaction_mask(scene.constraints, scene.witness_layout))  # witness solves it
        assert not all(satisfaction_mask(scene.constraints, scene.start_layout))  # not solved yet


def test_noise_free_solution_fits_in_six_moves():
    for seed in range(100):
        scene = scene_for_split("train", seed)
        state = {"constraints": scene.constraints, "layout": scene.start_layout, "budget": 8,
                 "undo_buffer": None, "start_budget": 8}
        actions, _ = plan(state)
        assert 1 <= len(actions) <= 6


def test_splits_share_no_templates():
    train = {template_key(scene_for_split("train", s).constraints) for s in range(400)}
    val = {template_key(s.constraints) for s in make_fixed_set("val", 100)}
    test = {template_key(s.constraints) for s in make_fixed_set("test", 100)}
    assert not (train & val) and not (train & test) and not (val & test)
    assert all(split_of_template(k) == "test" for k in test)


def test_fixed_sets_have_no_repeats_and_are_reproducible():
    a = make_fixed_set("test", 50)
    b = make_fixed_set("test", 50)
    assert a == b
    keys = {(template_key(s.constraints), s.start_layout) for s in a}
    assert len(keys) == 50
