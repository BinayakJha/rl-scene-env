"""All the experiment knobs in one place. run_all.py --quick swaps FULL for QUICK.

PPO hyperparameters were picked by looking at VALIDATION success only (never test).
See docs/code_guide.html, section "How PPO was tuned".
"""

FULL = {
    "seeds": [0, 1, 2],             # 3 independent training seeds per learned policy
    "val_scenes": 300,              # frozen validation set (model selection only)
    "test_scenes": 500,             # frozen test set (final numbers only)
    "train_check_scenes": 300,      # a frozen set of TRAINING scenes, to compare train vs val
    "test_rollouts": 4,             # each test scene is played 4 times (noise differs)
    "ppo_steps": 3_000_000,         # environment steps per PPO run
    "ppo_eval_every": 25,           # check validation every 25 updates (51,200 steps)
    "bc_demo_episodes": 6000,       # planner episodes used as demonstrations
    "bc_epochs": 60,
    "ablation_ppo_steps": 2_000_000,
    "parallel_jobs": 7,
}

QUICK = {
    "seeds": [0, 1, 2],
    "val_scenes": 60,
    "test_scenes": 80,
    "train_check_scenes": 60,
    "test_rollouts": 2,
    "ppo_steps": 61_440,
    "ppo_eval_every": 10,
    "bc_demo_episodes": 300,
    "bc_epochs": 5,
    "ablation_ppo_steps": 40_960,
    "parallel_jobs": 7,
}

# The three versions of the environment. "main" is the proposal's setting.
VARIANTS = {
    "main": {},                                # noise 0.7 / 0.2 / 0.1, UNDO allowed
    "deterministic": {"noise": (1.0, 0.0, 0.0)},  # ablation: every legal move succeeds
    "no_undo": {"allow_undo": False},          # ablation: UNDO switched off
}

PPO_HYPERPARAMETERS = {"learning_rate": 1e-3, "entropy_coef": 0.03}  # chosen on validation
HIDDEN_SIZE = 128
BC_LEARNING_RATE = 1e-3
BC_BATCH_SIZE = 256


def get(quick):
    return QUICK if quick else FULL


# ---- where files go (all relative to the project folder) ----
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
CURVES = RESULTS / "curves"
FIGURES = ROOT / "figures"


def results_dir(quick):
    """Quick runs write to results/quick/ so they never overwrite the real results."""
    return RESULTS / "quick" if quick else RESULTS
