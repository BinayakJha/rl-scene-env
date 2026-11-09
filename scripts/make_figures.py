"""Make every figure in figures/ from the saved results (matplotlib only, no AI images).

  learning_curves.png     PPO train and validation success vs environment steps (3 seeds)
  bc_curves.png           BC loss and validation success per epoch (3 seeds)
  success_bar.png         test full-success rate with 95% CIs, one bar per policy
  ablations_bar.png       the same for the deterministic and no-UNDO ablations
  worked_example.png      the proposal's worked example, step by step on the grid
  problem.png             one scene with its goals (used at the start of the guide)
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _setup  # noqa: F401
from scene_repair import settings
from scene_repair.drawing import draw_state
from scene_repair.env import STOP, SceneRepairEnv, action_name
from scene_repair.evaluation import evaluate, summarize
from scene_repair.policies.planner import PlannerPolicy
from scene_repair.scenes import load_scenes

POLICY_COLORS = {"random": "#9e9e9e", "bc": "#f2a541", "ppo": "#3d85c6", "planner": "#81b29a"}
POLICY_LABELS = {"random": "Random", "bc": "Behavior cloning", "ppo": "PPO", "planner": "Replanner"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def smooth(values, window=10):
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def learning_curves(out, figs, cfg, planner_val):
    fig, ax = plt.subplots(figsize=(7, 4))
    for n, seed in enumerate(cfg["seeds"]):
        data = json.load(open(out / "curves" / f"ppo_main_seed{seed}.json"))["curve"]
        steps = [p["steps"] for p in data]
        train = smooth([p["train_success"] for p in data])
        ax.plot(steps[len(steps) - len(train):], train, color="#3d85c6", alpha=0.35, lw=1,
                label="PPO train (smoothed)" if n == 0 else None)
        val = [(p["steps"], p["val_success"]) for p in data if "val_success" in p]
        ax.plot(*zip(*val), color="#1c4f86", lw=1.8, marker="o", ms=2.5,
                label="PPO validation" if n == 0 else None)
    ax.axhline(planner_val, color=POLICY_COLORS["planner"], ls="--", lw=1.5,
               label=f"Replanner on validation ({planner_val:.2f})")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("full-success rate")
    ax.set_ylim(0, 1.02)
    ax.set_title("PPO learning curves (3 seeds)")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "learning_curves.png", dpi=180)
    plt.close(fig)


def bc_curves(out, figs, cfg):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for seed in cfg["seeds"]:
        data = json.load(open(out / "curves" / f"bc_main_seed{seed}.json"))["curve"]
        epochs = [p["epoch"] for p in data]
        a1.plot(epochs, [p["loss"] for p in data], color="#f2a541", alpha=0.8)
        a2.plot(epochs, [p["val_success"] for p in data], color="#c47d0e", alpha=0.8)
    a1.set_title("BC training loss")
    a1.set_xlabel("epoch")
    a2.set_title("BC validation success")
    a2.set_xlabel("epoch")
    a2.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(figs / "bc_curves.png", dpi=180)
    plt.close(fig)


def bar_chart(ax, rows, title):
    order = [p for p in ("random", "bc", "ppo", "planner") if p in rows]
    for n, name in enumerate(order):
        s = rows[name]["pooled"]["success"]
        ax.bar(n, s["mean"], color=POLICY_COLORS[name], width=0.65)
        ax.errorbar(n, s["mean"], yerr=[[s["mean"] - s["ci_low"]], [s["ci_high"] - s["mean"]]],
                    color="black", capsize=4, lw=1)
        ax.text(n, s["ci_high"] + 0.02, f"{s['mean']:.2f}", ha="center", fontsize=9)
        for seed_row in rows[name].get("per_seed", []):  # one dot per training seed
            ax.scatter(n + 0.22, seed_row["success"]["mean"], color="black", s=10, zorder=3)
    ax.set_xticks(range(len(order)), [POLICY_LABELS[p] for p in order], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_title(title)


def success_bars(report, figs):
    fig, ax = plt.subplots(figsize=(6, 4))
    bar_chart(ax, report["variants"]["main"], "Test full-success rate (95% CI)")
    ax.set_ylabel("full-success rate")
    fig.tight_layout()
    fig.savefig(figs / "success_bar.png", dpi=180)
    plt.close(fig)


def ablation_bars(report, figs):
    names = [v for v in ("main", "deterministic", "no_undo") if v in report["variants"]]
    titles = {"main": "Main (noisy, UNDO on)", "deterministic": "No noise", "no_undo": "No UNDO"}
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.8), sharey=True)
    for ax, variant in zip(np.atleast_1d(axes), names):
        bar_chart(ax, report["variants"][variant], titles[variant])
        ax.tick_params(axis="x", labelsize=7.5)
    np.atleast_1d(axes)[0].set_ylabel("full-success rate")
    fig.tight_layout()
    fig.savefig(figs / "ablations_bar.png", dpi=180)
    plt.close(fig)


def worked_example(figs):
    """Replay the proposal's example with forced outcomes: fail, success, success, success, STOP."""
    import sys
    sys.path.insert(0, str(settings.ROOT / "tests"))
    from helpers import FAIL, SUCCESS, make_env_with, worked_example_scene
    env = make_env_with(worked_example_scene(), rolls=[FAIL, SUCCESS, SUCCESS, SUCCESS])
    actions = [2, 2, 4 + 3, 8 + 0, STOP]  # box E, box E, cat W, book N, STOP
    states, titles, reward = [env.state()], ["Start"], 0.0
    for a in actions:
        _, reward, _, _, info = env.step(a)
        states.append(info["state"])
        titles.append(f"{action_name(a, info['state']['names'])}\n→ {info['state']['outcome']}")
    titles[-1] += f"\nR = {reward:.3f}"
    fig, axes = plt.subplots(2, 3, figsize=(9, 8.2))
    for ax, st, title in zip(axes.flat, states, titles):
        draw_state(ax, st, title)
    fig.tight_layout()
    fig.savefig(figs / "worked_example.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.2, 4.0))
    draw_state(ax, states[0], "A scene: 3 objects, 3 goals")
    fig.tight_layout()
    fig.savefig(figs / "problem.png", dpi=200)
    plt.close(fig)


def main(quick):
    cfg = settings.get(quick)
    out = settings.results_dir(quick)
    figs = settings.FIGURES / "quick" if quick else settings.FIGURES
    figs.mkdir(parents=True, exist_ok=True)
    report = json.load(open(out / "test_results.json"))
    val = load_scenes(out / "data" / "val.json")
    planner_val = summarize(evaluate(SceneRepairEnv(), PlannerPolicy(), val, 1, base_seed=77))
    learning_curves(out, figs, cfg, planner_val["success"]["mean"])
    bc_curves(out, figs, cfg)
    success_bars(report, figs)
    ablation_bars(report, figs)
    worked_example(figs)
    print("figures written to", figs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    main(parser.parse_args().quick)
