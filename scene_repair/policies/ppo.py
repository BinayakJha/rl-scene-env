"""PPO (Proximal Policy Optimization), written out step by step.

One training loop ("update") does this:
  1. PLAY:   run 32 environments side by side for 64 steps each (2,048 steps), sampling
             actions from the current policy. Save obs, action, its probability, value guess,
             reward and done flag.
  2. SCORE:  for every step, work out the "advantage" = how much better things turned out
             than the value head expected. We use GAE (a smoothed version of
             "actual return minus predicted return"). gamma = 1 because the episode is short
             and only the final reward matters.
  3. LEARN:  for 4 passes over the batch, in mini-batches of 512:
               ratio = new probability of the action / old probability
               policy loss = -min(ratio * A, clip(ratio, 0.8, 1.2) * A)   <- the "proximal" part:
                             it stops one update from changing the policy too much
               value loss  = (value guess - actual return)^2
               entropy bonus keeps some randomness so it keeps exploring
  4. Repeat until the step budget is used up.
"""
from dataclasses import dataclass

import numpy as np
import torch

from ..env import SceneRepairEnv


@dataclass
class PPOConfig:
    total_steps: int = 1_500_000
    num_envs: int = 32
    steps_per_env: int = 64
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    anneal_lr: bool = True  # shrink the learning rate linearly to 0 by the end (steadier finish)


class VectorEnv:
    """Many environments stepped together. A finished one restarts at once on a new scene."""

    def __init__(self, num_envs, seed, env_kwargs):
        self.envs = [SceneRepairEnv(split="train", **env_kwargs) for _ in range(num_envs)]
        self.rng = np.random.default_rng(seed)
        self.obs, self.masks = [], []
        for env in self.envs:
            obs, info = env.reset(seed=self._new_seed())
            self.obs.append(obs)
            self.masks.append(info["action_mask"])

    def _new_seed(self):
        return int(self.rng.integers(0, 10**9))  # a random TRAINING scene seed

    def step(self, actions):
        rewards, dones, finished = [], [], []
        for n, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, done, _, info = env.step(action)
            if done:
                finished.append({"reward": reward, "success": all(info["state"]["mask"])})
                obs, info = env.reset(seed=self._new_seed())
            self.obs[n], self.masks[n] = obs, info["action_mask"]
            rewards.append(reward)
            dones.append(done)
        return np.array(rewards, dtype=np.float32), np.array(dones), finished


def compute_advantages(rewards, values, dones, last_values, gamma, lam):
    """GAE, walking backwards in time. Arrays are shaped [time, env]."""
    advantages = np.zeros_like(rewards)
    running = np.zeros(rewards.shape[1], dtype=np.float32)
    for t in reversed(range(len(rewards))):
        next_values = last_values if t == len(rewards) - 1 else values[t + 1]
        not_done = 1.0 - dones[t]  # after a finished episode, the future is worth 0
        delta = rewards[t] + gamma * next_values * not_done - values[t]
        running = delta + gamma * lam * not_done * running
        advantages[t] = running
    returns = advantages + values
    return advantages, returns


def ppo_update(net, optimizer, batch, cfg):
    """Step 3 above. Returns average losses and entropy, for the learning-curve logs."""
    obs, masks, actions, old_logp, advantages, returns = batch
    # normalise advantages: same signal, steadier step sizes
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    stats = []
    for _ in range(cfg.epochs):
        order = torch.randperm(len(actions))
        for start in range(0, len(order), cfg.minibatch_size):
            idx = order[start:start + cfg.minibatch_size]
            dist, value = net.distribution(obs[idx], masks[idx])
            logp = dist.log_prob(actions[idx])
            ratio = torch.exp(logp - old_logp[idx])
            unclipped = ratio * advantages[idx]
            clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * advantages[idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = ((value - returns[idx]) ** 2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            optimizer.step()
            stats.append([policy_loss.item(), value_loss.item(), entropy.item()])
    policy_loss, value_loss, entropy = np.mean(stats, axis=0)
    return {"policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy}


def train_ppo(net, cfg, seed, env_kwargs=None, on_update=None):
    """The whole PPO loop. `on_update(step_count, log)` is called after every update
    (the training script uses it to check validation and save the best network)."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
    venv = VectorEnv(cfg.num_envs, seed, env_kwargs or {})
    steps_done = 0
    T, N = cfg.steps_per_env, cfg.num_envs
    while steps_done < cfg.total_steps:
        if cfg.anneal_lr:
            for group in optimizer.param_groups:
                group["lr"] = cfg.learning_rate * (1 - steps_done / cfg.total_steps)
        # ---- 1. PLAY
        obs_buf = np.zeros((T, N, len(venv.obs[0])), dtype=np.float32)
        mask_buf = np.zeros((T, N, len(venv.masks[0])), dtype=bool)
        act_buf = np.zeros((T, N), dtype=np.int64)
        logp_buf = np.zeros((T, N), dtype=np.float32)
        val_buf = np.zeros((T, N), dtype=np.float32)
        rew_buf = np.zeros((T, N), dtype=np.float32)
        done_buf = np.zeros((T, N), dtype=np.float32)
        finished = []
        for t in range(T):
            obs_buf[t], mask_buf[t] = np.array(venv.obs), np.array(venv.masks)
            with torch.no_grad():
                dist, value = net.distribution(torch.as_tensor(obs_buf[t]), torch.as_tensor(mask_buf[t]))
                action = dist.sample()
            act_buf[t], logp_buf[t], val_buf[t] = action.numpy(), dist.log_prob(action).numpy(), value.numpy()
            rew_buf[t], done_buf[t], ended = venv.step(action.numpy())
            finished += ended
        steps_done += T * N

        # ---- 2. SCORE
        with torch.no_grad():
            _, last_values = net.distribution(torch.as_tensor(np.array(venv.obs)),
                                              torch.as_tensor(np.array(venv.masks)))
        adv, ret = compute_advantages(rew_buf, val_buf, done_buf, last_values.numpy(),
                                      cfg.gamma, cfg.gae_lambda)

        # ---- 3. LEARN (flatten [time, env] into one long batch first)
        flat = lambda a: torch.as_tensor(a.reshape(T * N, *a.shape[2:]))
        net.train()
        log = ppo_update(net, optimizer, (flat(obs_buf), flat(mask_buf), flat(act_buf),
                                          flat(logp_buf), flat(adv), flat(ret)), cfg)
        net.eval()
        log["train_success"] = float(np.mean([f["success"] for f in finished])) if finished else 0.0
        log["train_reward"] = float(np.mean([f["reward"] for f in finished])) if finished else 0.0
        log["episodes"] = len(finished)
        if on_update:
            on_update(steps_done, log)
    return steps_done
