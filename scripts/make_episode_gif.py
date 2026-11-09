"""Animate one real test episode played by PPO (seed 0) as a GIF plus a still strip.

Which episode: the first test scene (in file order) where a move went wrong (jitter or
fail) and PPO still reached full success. That shows recovery, which is the point of the task.
"""
import argparse
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import _setup  # noqa: F401
from scene_repair import settings
from scene_repair.drawing import draw_state
from scene_repair.env import SceneRepairEnv, action_name
from scene_repair.evaluation import run_episode
from scene_repair.networks import NetPolicy
from scene_repair.scenes import load_scenes
from evaluate import TEST_SEED, load_net


def pick_episode(env, policy, scenes):
    for n, scene in enumerate(scenes):
        record = run_episode(env, policy, scene, seed=TEST_SEED + n * 1000)
        outcomes = {t["outcome"] for t in record["trajectory"]}
        if record["success"] and outcomes & {"jitter", "fail"}:
            return n, record
    return 0, run_episode(env, policy, scenes[0], seed=TEST_SEED)


def frame_titles(trajectory):
    titles = ["Start"]
    for before, after in zip(trajectory, trajectory[1:]):
        titles.append(f"{action_name(before['action'], before['names'])} → {after['outcome']}")
    return titles


def main(quick):
    out = settings.results_dir(quick)
    figs = settings.FIGURES / "quick" if quick else settings.FIGURES
    scenes = load_scenes(out / "data" / "test.json")
    policy = NetPolicy(load_net(out / "models" / "ppo_main_seed0.pt"), "ppo")
    n, record = pick_episode(SceneRepairEnv(), policy, scenes)
    traj, titles = record["trajectory"], frame_titles(record["trajectory"])
    titles[-1] += f"   (R = {record['reward']:.3f})"

    frames = []
    for state, title in zip(traj, titles):
        fig, ax = plt.subplots(figsize=(3.6, 4.4))
        draw_state(ax, state, title)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110)
        plt.close(fig)
        frames.append(Image.open(buffer).convert("RGB"))
    frames[0].save(figs / "ppo_episode.gif", save_all=True, append_images=frames[1:] + [frames[-1]],
                   duration=1100, loop=0)

    fig, axes = plt.subplots(1, len(traj), figsize=(2.7 * len(traj), 3.9))
    for ax, state, title in zip(axes, traj, titles):
        draw_state(ax, state, title.replace(" → ", "\n→ ").replace("   (", "\n("))
    fig.suptitle(f"PPO (seed 0) on test scene #{n}", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / "ppo_episode.png", dpi=170)
    plt.close(fig)
    print(f"episode from test scene {n}: {len(traj) - 1} steps, R = {record['reward']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    main(parser.parse_args().quick)
