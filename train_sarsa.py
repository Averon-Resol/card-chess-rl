"""Member 3 (Dano): Train & Evaluate SARSA at Move Depth 6.

Usage:
    python train_sarsa.py                        # Full run (25 iters x 16 games @ depth 6) + 90-game evaluation
    python train_sarsa.py --iters 5 --games 4    # Quick test run

Outputs (commit these!):
    results/sarsa_results.json   full evaluation details
    results/score_sarsa.txt      one appended line: the score used for model selection
    models/sarsa.pt              the trained weights
"""
import argparse
import random

import numpy as np
import torch

from cardchess_rl.agent.sarsa_trainer import SARSATrainer


def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate SARSA at Depth 6 (Owner: Dano)")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--max-rounds", type=int, default=60)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("  CARD-CHESS RL — ALGORITHM 3: SARSA (Owner: Dano)")
    print(f"  Move Depth: {args.depth} (Stockfish) | Iterations: {args.iters} | Games/Iter: {args.games}")
    print("=" * 72, flush=True)

    trainer = SARSATrainer(
        games_per_iter=args.games, move_depth=args.depth, max_rounds=args.max_rounds,
        use_stockfish=True, save_dir="checkpoints/sarsa",
    )
    try:
        trainer.train(n_iterations=args.iters, eval_interval=max(1, args.iters // 5),
                      log_file="results/sarsa_training_log.json")
        res = trainer.evaluate_full_suite(n_games_per_opponent=args.eval_games, output_dir="results")
        print(f"SARSA Final Overall Accuracy: {res['overall_accuracy'] * 100:.2f}%")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
