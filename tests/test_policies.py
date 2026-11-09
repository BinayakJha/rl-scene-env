"""Policies: the planner solves the worked example; nets respect the action mask; PPO math."""
import numpy as np
import torch

from helpers import SUCCESS, make_env_with, worked_example_scene
from scene_repair.encoding import OBS_SIZE, encode
from scene_repair.env import NUM_ACTIONS, STOP
from scene_repair.networks import NetPolicy, PolicyValueNet
from scene_repair.policies.planner import PlannerPolicy, plan
from scene_repair.policies.ppo import compute_advantages
from scene_repair.policies.random_policy import RandomPolicy


def test_planner_finds_the_three_move_fix_for_the_worked_example():
    env = make_env_with(worked_example_scene(), rolls=[SUCCESS] * 3)
    actions, _ = plan(env.state())
    assert len(actions) == 3
    planner, info = PlannerPolicy(), {"state": env.state()}
    for _ in range(3):
        _, _, _, _, info = env.step(planner.act(None, info))
    assert all(info["state"]["mask"])
    assert planner.act(None, info) == STOP


def test_random_policy_only_picks_allowed_actions():
    env = make_env_with(worked_example_scene())
    policy = RandomPolicy(seed=0)
    info = {"action_mask": env.action_mask()}
    picks = {policy.act(None, info) for _ in range(500)}
    assert all(info["action_mask"][a] for a in picks)


def test_network_never_picks_a_masked_action():
    env = make_env_with(worked_example_scene())
    net = PolicyValueNet(OBS_SIZE, NUM_ACTIONS)
    policy = NetPolicy(net, "test", greedy=False)
    obs, info = encode(env.state()), {"action_mask": env.action_mask()}
    for _ in range(200):
        assert info["action_mask"][policy.act(obs, info)]


def test_encoding_has_fixed_size():
    env = make_env_with(worked_example_scene())
    assert encode(env.state()).shape == (OBS_SIZE,)


def test_advantages_with_gamma_one_equal_final_reward_minus_value():
    # one env, 3 steps, reward only at the end; lambda = 1 -> advantage = R - V
    rewards = np.array([[0.0], [0.0], [0.9]], dtype=np.float32)
    values = np.array([[0.5], [0.6], [0.7]], dtype=np.float32)
    dones = np.array([[0.0], [0.0], [1.0]], dtype=np.float32)
    adv, ret = compute_advantages(rewards, values, dones, np.array([0.3]), 1.0, 1.0)
    assert np.allclose(ret[:, 0], 0.9)
    assert np.allclose(adv[:, 0], 0.9 - values[:, 0])
