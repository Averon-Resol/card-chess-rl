"""Model selection for the GUI / final project.

Each team member's training script appends their final accuracy to a text file
in `results/` and saves their trained weights in `models/`:

  - Member 1 (Ihsal): `python train_ppo.py`   -> appends to `results/score_ppo.txt`,   saves `models/ppo.pt`
  - Member 2 (Dhruv): `python train_dqn.py`   -> appends to `results/score_dqn.txt`,   saves `models/dqn.pt`
  - Member 3 (Dano):  `python train_sarsa.py` -> appends to `results/score_sarsa.txt`, saves `models/sarsa.pt`

Below, variables `a`, `b`, and `c` store the accuracy of PPO, DQN, and SARSA
read from those text files (or you can manually type numbers for `a`, `b`, `c`
if you want to override them by hand). The code takes `max(a, b, c)` to decide
which model the site (`cardchess_rl/gui/app.py`) loads.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

ALGOS = ("ppo", "dqn", "sarsa")


def read_score(algo: str) -> float | None:
    """Read the latest appended accuracy for `algo` from `results/score_<algo>.txt`."""
    path = RESULTS_DIR / f"score_{algo}.txt"
    if not path.exists():
        return None
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return float(lines[-1].split(",")[0])
    except ValueError:
        return None


# =========================================================================== #
#  MODEL SELECTION VARIABLES (a = PPO, b = DQN, c = SARSA)
#  By default these are read automatically from the appended text files.
#  You can also manually replace any of them with a number (e.g. a = 0.72)
#  or set FORCE_MODEL = "ppo" / "dqn" / "sarsa" to force a specific model.
# =========================================================================== #
FORCE_MODEL: str | None = None

a: float = read_score("ppo") or -1.0    # Ihsal — PPO accuracy
b: float = read_score("dqn") or -1.0    # Dhruv — Double DQN accuracy
c: float = read_score("sarsa") or -1.0  # Dano  — SARSA accuracy


def read_scores() -> dict[str, float | None]:
    """Return the latest scores dictionary {'ppo': a, 'dqn': b, 'sarsa': c}."""
    cur_a = read_score("ppo")
    cur_b = read_score("dqn")
    cur_c = read_score("sarsa")
    return {
        "ppo": cur_a if cur_a is not None else (a if a >= 0 else None),
        "dqn": cur_b if cur_b is not None else (b if b >= 0 else None),
        "sarsa": cur_c if cur_c is not None else (c if c >= 0 else None),
    }


def best_model() -> tuple[str, Path] | None:
    """Compare variables (a, b, c) via max() and return (algo_name, model_path)."""
    scores = read_scores()
    score_a = scores["ppo"] if scores["ppo"] is not None else a
    score_b = scores["dqn"] if scores["dqn"] is not None else b
    score_c = scores["sarsa"] if scores["sarsa"] is not None else c

    candidates = {
        "ppo": score_a,
        "dqn": score_b,
        "sarsa": score_c,
    }

    if FORCE_MODEL is not None:
        choice = FORCE_MODEL.lower()
    else:
        # Filter to models that have been trained (score >= 0 and weights file exists)
        valid = {
            algo: val
            for algo, val in candidates.items()
            if val is not None and val >= 0.0 and (MODELS_DIR / f"{algo}.pt").exists()
        }
        if not valid:
            return None
        choice = max(valid, key=valid.get)

    path = MODELS_DIR / f"{choice}.pt"
    return (choice, path) if path.exists() else None


if __name__ == "__main__":
    scores = read_scores()
    print(f"a (PPO) = {scores['ppo']} | b (DQN) = {scores['dqn']} | c (SARSA) = {scores['sarsa']}")
    print("Selected model:", best_model())
