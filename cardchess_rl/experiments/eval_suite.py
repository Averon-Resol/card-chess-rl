"""Standardized depth-6 evaluation suite for comparing PPO, DQN, and A2C (Revision 2).

Every algorithm is evaluated against the three baseline card-pickers:
  1. Random picker
  2. Always-first picker
  3. Greedy 1-ply heuristic picker

To make the comparison across all three group members 100% fair:
  - Chess moves for both sides are played by Stockfish at depth 6.
  - Game `i` uses a fixed random seed (`base_seed + i`) and alternates agent color
    (even `i`: White, odd `i`: Black).
  - If a game reaches `max_rounds` (60) without checkmate/draw, it is adjudicated
    by Stockfish depth-6 centipawn evaluation (> +50 cp = win, < -50 cp = loss, else draw).
  - Accuracy / Score is defined as:
        accuracy = (wins + 0.5 * draws) / n_games
"""
from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime
from typing import Callable, Dict

import chess
import torch

from cardchess_rl.engine.game import CardChessGame
from cardchess_rl.experiments.baselines import (
    always_first_picker,
    greedy_picker,
    random_picker,
    trained_picker,
)
from cardchess_rl.search.stockfish_engine import StockfishMover


def play_eval_game(
    agent_picker: Callable,
    opponent_picker: Callable,
    agent_color: chess.Color,
    mover: StockfishMover,
    max_rounds: int = 60,
    seed: int = 42,
) -> Dict[str, float | int | str]:
    """Play a single depth-6 evaluation game with a deterministic seed."""
    py_state = random.getstate()
    random.seed(seed)
    try:
        game = CardChessGame()

        def card_picker(state, candidates):
            if state.picker == agent_color:
                return agent_picker(state, candidates)
            return opponent_picker(state, candidates)

        def move_maker(_state, moves):
            return mover.choose_move(game.board, moves)

        while not game.is_game_over() and game.round_number <= max_rounds:
            game.play_round(card_picker, move_maker, move_maker)

        if game.is_game_over() and game.result is not None:
            white_res = game.result
            outcome_type = "terminal"
        else:
            # Adjudicate capped game using Stockfish depth-6 evaluation from White's POV
            cp_white = mover.score_cp(game.board, chess.WHITE, depth=mover.depth)
            if cp_white > 50.0:
                white_res = 1.0
            elif cp_white < -50.0:
                white_res = -1.0
            else:
                white_res = 0.0
            outcome_type = "adjudicated"

        agent_res = white_res if agent_color == chess.WHITE else -white_res
        return {
            "agent_result": float(agent_res),  # +1 win, 0 draw, -1 loss
            "rounds": int(game.round_number - 1),
            "outcome_type": outcome_type,
        }
    finally:
        random.setstate(py_state)


def evaluate_against_opponent(
    agent_picker: Callable,
    opponent_picker: Callable,
    opponent_name: str,
    mover: StockfishMover,
    n_games: int = 30,
    max_rounds: int = 60,
    base_seed: int = 10_000,
) -> Dict[str, float | int | str]:
    """Evaluate `agent_picker` against a single baseline over `n_games` seeded games."""
    wins, draws, losses = 0, 0, 0
    total_rounds = 0
    t0 = time.time()

    for i in range(n_games):
        agent_color = chess.WHITE if (i % 2 == 0) else chess.BLACK
        res = play_eval_game(
            agent_picker=agent_picker,
            opponent_picker=opponent_picker,
            agent_color=agent_color,
            mover=mover,
            max_rounds=max_rounds,
            seed=base_seed + i,
        )
        if res["agent_result"] > 0:
            wins += 1
        elif res["agent_result"] < 0:
            losses += 1
        else:
            draws += 1
        total_rounds += res["rounds"]

    score_accuracy = (wins + 0.5 * draws) / n_games if n_games > 0 else 0.0
    win_rate_strict = wins / n_games if n_games > 0 else 0.0

    return {
        "opponent": opponent_name,
        "n_games": n_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "accuracy_score": float(score_accuracy),
        "strict_win_rate": float(win_rate_strict),
        "avg_rounds": float(total_rounds / max(1, n_games)),
        "elapsed_sec": round(time.time() - t0, 2),
    }


def evaluate_all_baselines(
    net: torch.nn.Module,
    algo_name: str,
    member_name: str,
    mover: StockfishMover,
    device: str | torch.device = "cpu",
    n_games_per_opponent: int = 30,
    max_rounds: int = 60,
    checkpoint_path: str = "",
    training_summary: Dict | None = None,
    output_dir: str = "results",
) -> Dict:
    """Run the full 3-baseline evaluation suite and save `results/<algo>_results.json`."""
    os.makedirs(output_dir, exist_ok=True)
    picker = trained_picker(net, device=str(device))

    opponents = [
        ("Random", random_picker, 10_000),
        ("Always-First", always_first_picker, 20_000),
        ("Greedy-1Ply", greedy_picker, 30_000),
    ]

    matchups = {}
    for opp_name, opp_fn, seed_offset in opponents:
        print(f"  [Eval] {algo_name.upper()} vs {opp_name} ({n_games_per_opponent} games at depth {mover.depth})...", flush=True)
        res = evaluate_against_opponent(
            agent_picker=picker,
            opponent_picker=opp_fn,
            opponent_name=opp_name,
            mover=mover,
            n_games=n_games_per_opponent,
            max_rounds=max_rounds,
            base_seed=seed_offset,
        )
        matchups[opp_name] = res
        print(
            f"         W/D/L = {res['wins']}/{res['draws']}/{res['losses']} | "
            f"Accuracy (Score) = {res['accuracy_score']*100:.1f}% | "
            f"Strict Win Rate = {res['strict_win_rate']*100:.1f}% ({res['elapsed_sec']}s)",
            flush=True,
        )

    overall_acc = sum(m["accuracy_score"] for m in matchups.values()) / len(matchups)
    overall_strict = sum(m["strict_win_rate"] for m in matchups.values()) / len(matchups)

    payload = {
        "algorithm": algo_name.upper(),
        "member": member_name,
        "move_depth": mover.depth,
        "n_games_per_opponent": n_games_per_opponent,
        "max_rounds": max_rounds,
        "overall_accuracy": float(overall_acc),
        "overall_strict_win_rate": float(overall_strict),
        "matchups": matchups,
        "checkpoint_path": checkpoint_path,
        "training_summary": training_summary or {},
        "timestamp": str(datetime.now()),
    }

    out_file = os.path.join(output_dir, f"{algo_name.lower()}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Per-algorithm score file (appended, so history is kept). One file per
    # algorithm means three people can push without git merge conflicts.
    score_file = os.path.join(output_dir, f"score_{algo_name.lower()}.txt")
    with open(score_file, "a", encoding="utf-8") as f:
        f.write(f"{overall_acc:.6f},{datetime.now().isoformat(timespec='seconds')},{member_name}\n")

    # Trained weights go to models/ (NOT gitignored) so the GUI can load them.
    models_dir = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{algo_name.lower()}.pt")
    torch.save({"algo": algo_name.upper(), "model_state_dict": net.state_dict()}, model_path)
    print(f"  Saved score to {score_file} and weights to {model_path}")

    print("\n" + "=" * 72)
    print(f"  FINAL RESULTS — {algo_name.upper()} (Owner: {member_name}) | Move Depth: {mover.depth}")
    print("=" * 72)
    print(f"  {'Opponent':<16} {'W / D / L':<14} {'Accuracy (W+0.5D)':<20} {'Strict Win %':<14}")
    print("-" * 72)
    for opp_name, m in matchups.items():
        wdl = f"{m['wins']}/{m['draws']}/{m['losses']}"
        print(
            f"  {opp_name:<16} {wdl:<14} "
            f"{m['accuracy_score']*100:>6.2f}%              "
            f"{m['strict_win_rate']*100:>6.2f}%"
        )
    print("-" * 72)
    print(
        f"  {'OVERALL MEAN':<16} {'':<14} "
        f"{overall_acc*100:>6.2f}%              "
        f"{overall_strict*100:>6.2f}%"
    )
    print("=" * 72)
    print(f"  Saved results to: {out_file}\n", flush=True)
    return payload
