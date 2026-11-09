"""Reproduce EVERYTHING with one command.

    python run_all.py           full run (about 18 minutes on 8 CPU cores)
    python run_all.py --quick   tiny version (about 1 minute) to check that it all works;
                                writes to results/quick/ and figures/quick/

Steps: unit tests -> freeze splits -> train BC and PPO (3 seeds x 3 variants, several at
once) -> evaluate on the test set -> make figures and the episode GIF.
"""
import argparse
import json
import subprocess
import sys
import time

from scene_repair import settings

PYTHON = sys.executable


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_parallel(commands, max_jobs):
    """Run several training jobs at the same time (each uses one CPU core)."""
    waiting, running = list(commands), []
    while waiting or running:
        while waiting and len(running) < max_jobs:
            cmd = waiting.pop(0)
            print("$", " ".join(cmd), flush=True)
            running.append(subprocess.Popen(cmd))
        time.sleep(2)
        for proc in list(running):
            if proc.poll() is not None:
                if proc.returncode != 0:
                    raise SystemExit(f"job failed: {proc.args}")
                running.remove(proc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    quick = parser.parse_args().quick
    skip_tests = parser.parse_args().skip_tests
    flag = ["--quick"] if quick else []
    cfg = settings.get(quick)
    timings, start = {}, time.time()

    if not skip_tests:
        run([PYTHON, "-m", "pytest", "-q", "tests"])
    timings["tests"] = time.time() - start

    run([PYTHON, "scripts/make_splits.py", *flag])
    run([PYTHON, "scripts/scene_stats.py", *flag])

    t = time.time()
    jobs = []
    for variant in settings.VARIANTS:  # main first, then the two ablations
        for seed in cfg["seeds"]:
            jobs.append([PYTHON, "scripts/train_ppo.py", "--seed", str(seed), "--variant", variant, *flag])
            jobs.append([PYTHON, "scripts/train_bc.py", "--seed", str(seed), "--variant", variant, *flag])
    run_parallel(jobs, cfg["parallel_jobs"])
    timings["training"] = time.time() - t

    t = time.time()
    run([PYTHON, "scripts/evaluate.py", *flag])
    run([PYTHON, "scripts/failure_analysis.py", *flag])
    timings["evaluation"] = time.time() - t

    t = time.time()
    run([PYTHON, "scripts/make_figures.py", *flag])
    run([PYTHON, "scripts/make_episode_gif.py", *flag])
    timings["figures"] = time.time() - t
    timings["total"] = time.time() - start

    with open(settings.results_dir(quick) / "run_log.json", "w") as f:
        json.dump({"quick": quick, "seconds": timings, "settings": cfg}, f, indent=1)
    print("\nDone in", round(timings["total"] / 60, 1), "minutes. See results/ and figures/.")


if __name__ == "__main__":
    main()
