"""Print the PPO / DQN / SARSA leaderboard and which model the app will use.

Usage:
    python compare_results.py

Reads results/<algo>_results.json for detail and cardchess_rl/model_selector.py for the
selection rule (max of the latest scores A/B/C, unless FORCE_MODEL is set).
"""
import json
import os

from cardchess_rl.model_selector import ALGOS, best_model, read_scores

NAMES = {"ppo": "PPO", "dqn": "DQN", "sarsa": "SARSA"}


def main():
    scores = read_scores()
    print("\n" + "=" * 84)
    print("  CARD-CHESS RL — 3-ALGORITHM DEPTH-6 COMPARISON")
    print("=" * 84)
    print(f"  {'Algo':<8} {'Member':<8} {'vs Random':<12} {'vs First':<12} {'vs Greedy':<12} {'OVERALL (score file)':<20}")
    print("-" * 84)
    for algo in sorted(ALGOS, key=lambda a: -(scores[a] if scores[a] is not None else -1)):
        path = os.path.join("results", f"{algo}_results.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                r = json.load(f)
            m = r.get("matchups", {})
            cells = [m.get(k, {}).get("accuracy_score", 0.0) * 100 for k in ("Random", "Always-First", "Greedy-1Ply")]
            member = r.get("member", "?")
        else:
            cells, member = [None] * 3, "-"
        fmt = lambda v: f"{v:6.2f}%" if v is not None else "  n/a  "
        overall = f"{scores[algo] * 100:6.2f}%" if scores[algo] is not None else "not trained"
        print(f"  {NAMES[algo]:<8} {member:<8} {fmt(cells[0]):<12} {fmt(cells[1]):<12} {fmt(cells[2]):<12} {overall:<20}")
    print("=" * 84)
    choice = best_model()
    if choice:
        print(f"  The app will use: {NAMES[choice[0]]}  ({choice[1]})")
    else:
        print("  No trained model available yet -> app falls back to the Greedy heuristic.")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
