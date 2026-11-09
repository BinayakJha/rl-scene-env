"""Playing episodes and summarising them: success rate, mean reward, actions used, and CIs.

CI = 95% confidence interval. We get it by "bootstrapping over test scenes": resample the
test scenes with replacement 2,000 times, recompute the average each time, and keep the
middle 95% of those averages. Scenes are resampled (not single episodes) because the
episodes of one scene are not independent of each other.
"""
import numpy as np

from .constraints import fraction_satisfied


def run_episode(env, policy, scene, seed):
    """Play one episode of `scene`. Returns a small record, including the full trajectory."""
    obs, info = env.reset(seed=seed, options={"scene": scene})
    trajectory = [dict(info["state"], action=None)]
    done, reward = False, 0.0
    while not done:
        action = policy.act(obs, info)
        obs, reward, done, _, info = env.step(action)
        trajectory[-1]["action"] = action
        trajectory.append(dict(info["state"], action=None))
    final = info["state"]
    return {
        "success": bool(all(final["mask"])),
        "reward": float(reward),
        "s": fraction_satisfied(final["constraints"], final["layout"]),
        "actions_used": env.start_budget - final["budget"],  # billed actions (STOP is free)
        "steps": len(trajectory) - 1,
        "undos": sum(1 for t in trajectory[1:] if t["outcome"] in ("undo", "undo_empty")),
        "trajectory": trajectory,
    }


def evaluate(env, policy, scenes, rollouts_per_scene, base_seed=0, keep_trajectories=False):
    """Play every scene `rollouts_per_scene` times. Rollout r of scene n always uses the same seed."""
    records = []
    for n, scene in enumerate(scenes):
        for r in range(rollouts_per_scene):
            record = run_episode(env, policy, scene, seed=base_seed + n * 1000 + r)
            record["scene_index"] = n
            if not keep_trajectories:
                del record["trajectory"]
            records.append(record)
    return records


def bootstrap_ci(records, key, num_resamples=2000, seed=0):
    """Mean of `key` and its 95% CI, resampling whole scenes."""
    by_scene = {}
    for r in records:
        by_scene.setdefault(r["scene_index"], []).append(float(r[key]))
    per_scene = np.array([np.mean(v) for v in by_scene.values()])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(per_scene), size=(num_resamples, len(per_scene)))
    means = per_scene[picks].mean(axis=1)
    return float(per_scene.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(records):
    """The headline numbers for one batch of episodes."""
    out = {"episodes": len(records)}
    for key in ("success", "reward", "s", "actions_used", "undos"):
        mean, low, high = bootstrap_ci(records, key)
        out[key] = {"mean": mean, "ci_low": low, "ci_high": high}
    return out
