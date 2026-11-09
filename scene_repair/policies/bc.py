"""Behavior cloning (BC): copy the planner.

  1. Let the planner play training scenes WITH noise. Because moves fail and jitter, the
     planner also visits messy "recovery" states, so the demos are not just perfect paths.
  2. Save every (observation, action the planner chose) pair.
  3. Train the network as a classifier: given the observation, predict the planner's action
     (cross-entropy loss). It is "reactive": it looks only at the current state.
"""
import numpy as np
import torch

from ..env import SceneRepairEnv
from .planner import PlannerPolicy


def collect_demos(num_episodes, env_kwargs=None, seed_offset=0):
    """Planner demos on training scenes. Returns observations, masks, actions and step count."""
    env = SceneRepairEnv(split="train", **(env_kwargs or {}))
    planner = PlannerPolicy(allow_undo=env.allow_undo)
    observations, masks, actions = [], [], []
    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed_offset + episode)  # a fresh training scene
        done = False
        while not done:
            action = planner.act(obs, info)
            observations.append(obs)
            masks.append(info["action_mask"])
            actions.append(action)
            obs, _, done, _, info = env.step(action)
    return (np.array(observations), np.array(masks), np.array(actions), planner.total_expansions)


def train_bc_epoch(net, optimizer, observations, masks, actions, batch_size, rng):
    """One pass over the demos in random order. Returns the average loss."""
    order = rng.permutation(len(actions))
    losses = []
    net.train()
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        logits, _ = net(torch.as_tensor(observations[idx]), torch.as_tensor(masks[idx]))
        loss = torch.nn.functional.cross_entropy(logits, torch.as_tensor(actions[idx]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    net.eval()
    return float(np.mean(losses))
