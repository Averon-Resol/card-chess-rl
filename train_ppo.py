"""Member 1 (Ihsal): Train & Evaluate PPO at Move Depth 6.

Usage:
    python train_ppo.py                          # Full training (25 iters x 16 games @ depth 6) + 90-game evaluation
    python train_ppo.py --iters 5 --games 4      # Quick test run
"""
import argparse
import random
import numpy as np
import torch

from cardchess_rl.agent.training import PPOTrainer


def main():
    parser = argparse.ArgumentParser(description="Train & Evaluate PPO at Depth 6 (Owner: Ihsal)")
    parser.add_argument("--iters", type=int, default=25, help="Number of training iterations (default: 25)")
    parser.add_argument("--games", type=int, default=16, help="Self-play games per iteration (default: 16)")
    parser.add_argument("--depth", type=int, default=6, help="Stockfish move search depth (default: 6)")
    parser.add_argument("--max-rounds", type=int, default=60, help="Max rounds per game before adjudication (default: 60)")
    parser.add_argument("--eval-games", type=int, default=30, help="Evaluation games per baseline opponent (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 72)
    print("  CARD-CHESS RL — ALGORITHM 1: PPO (Owner: Ihsal)")
    print(f"  Move Depth: {args.depth} (Stockfish) | Iterations: {args.iters} | Games/Iter: {args.games}")
    print("=" * 72, flush=True)

    trainer = PPOTrainer(
        games_per_iter=args.games,
        move_depth=args.depth,
        max_rounds=args.max_rounds,
        use_stockfish=True,
        save_dir="checkpoints/ppo",
    )
    try:
        trainer.train(
            n_iterations=args.iters,
            eval_interval=max(1, args.iters // 5),
            log_file="results/ppo_training_log.json",
        )
        res = trainer.evaluate_full_suite(
            n_games_per_opponent=args.eval_games,
            output_dir="results",
        )
        print(f"PPO Final Overall Accuracy: {res['overall_accuracy'] * 100:.2f}%")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
