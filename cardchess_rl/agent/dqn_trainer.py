import json
import os
import random
from collections import deque
from datetime import datetime
from typing import Optional

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cardchess_rl.agent.env import CardChessEnv
from cardchess_rl.agent.network import QNet
from cardchess_rl.experiments.eval_suite import evaluate_all_baselines
from cardchess_rl.search.stockfish_engine import StockfishMover


class DQNTrainer:
    """Double Deep Q-Network (Double DQN) trainer for Card-Chess (Owner: Dhruv).

    Key features:
      - Identical Conv2D + FC trunk (`QNet`) to `PolicyValueNet` (~1.13M parameters).
      - Experience Replay buffer (`replay_size=20000`) + periodic Target Network sync.
      - SMDP Bellman target using variable-interval discounting:
            y = r + (1 - done) * (gamma ** dt) * Q_target(s', argmax_a' Q_online(s', a'))
      - Epsilon-greedy exploration decaying linearly from `eps_start=1.0` to `eps_end=0.05`.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        replay_size: int = 20_000,
        batch_size: int = 64,
        updates_per_iter: int = 16,
        target_sync_interval: int = 2,
        games_per_iter: int = 16,
        move_depth: int = 6,
        max_rounds: int = 60,
        use_stockfish: bool = True,
        save_dir: str = "checkpoints/dqn",
    ):
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.epsilon = eps_start
        self.batch_size = batch_size
        self.updates_per_iter = updates_per_iter
        self.target_sync_interval = target_sync_interval
        self.games_per_iter = games_per_iter
        self.move_depth = move_depth
        self.max_rounds = max_rounds
        self.use_stockfish = use_stockfish
        self.save_dir = save_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = QNet().to(self.device)
        self.target_net = QNet().to(self.device)
        self.target_net.load_state_dict(self.net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.replay = deque(maxlen=replay_size)
        self.mover: Optional[StockfishMover] = (
            StockfishMover(depth=self.move_depth) if self.use_stockfish else None
        )

        os.makedirs(self.save_dir, exist_ok=True)

    def close(self) -> None:
        if self.mover is not None:
            self.mover.close()
            self.mover = None

    def select_action(self, state: dict, greedy: bool = False) -> int:
        if (not greedy) and (random.random() < self.epsilon):
            return random.randint(0, 2)
        board_t = state["board"].detach().clone().unsqueeze(0).to(self.device)
        cards_t = state["cards"].detach().clone().unsqueeze(0).to(self.device)
        deck_t = state["deck"].detach().clone().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_vals = self.net(board_t, cards_t, deck_t)
            return int(torch.argmax(q_vals, dim=-1).item())

    def collect_games(self, n_games: int) -> list[dict]:
        """Play `n_games` episodes with epsilon-greedy exploration and push to replay buffer."""
        self.net.eval()
        episode_stats = []

        for game_idx in range(n_games):
            color = chess.WHITE if (game_idx % 2 == 0) else chess.BLACK
            env = CardChessEnv(
                agent_color=color,
                move_depth=self.move_depth,
                gamma=self.gamma,
                max_rounds=self.max_rounds,
                use_stockfish=self.use_stockfish,
                mover=self.mover,
            )
            state = env.reset()
            done = False
            ep_reward = 0.0
            terminal_rew = 0.0

            while not done:
                action = self.select_action(state, greedy=False)
                next_state, reward, done, info = env.step(action)
                dt = int(info.get("dt", 1))

                self.replay.append(
                    (
                        state["board"].numpy(),
                        state["cards"].numpy(),
                        state["deck"].numpy(),
                        int(action),
                        float(reward),
                        next_state["board"].numpy(),
                        next_state["cards"].numpy(),
                        next_state["deck"].numpy(),
                        float(done),
                        dt,
                    )
                )
                ep_reward += float(reward)
                if done:
                    terminal_rew = float(info.get("terminal_reward", 0.0))
                state = next_state

            episode_stats.append({"ep_reward": ep_reward, "terminal_reward": terminal_rew})

        return episode_stats

    def dqn_update(self) -> float:
        """Sample mini-batches from replay buffer and run Double DQN updates."""
        if len(self.replay) < max(16, self.batch_size // 2):
            return 0.0

        self.net.train()
        losses = []
        loss_fn = nn.SmoothL1Loss()

        for _ in range(self.updates_per_iter):
            k = min(len(self.replay), self.batch_size)
            batch = random.sample(self.replay, k)
            b_s, c_s, d_s, acts, rews, nb_s, nc_s, nd_s, dones, dts = zip(*batch)

            board_t = torch.tensor(np.array(b_s), dtype=torch.float32, device=self.device)
            cards_t = torch.tensor(np.array(c_s), dtype=torch.float32, device=self.device)
            deck_t = torch.tensor(np.array(d_s), dtype=torch.float32, device=self.device)
            actions_t = torch.tensor(acts, dtype=torch.int64, device=self.device).unsqueeze(1)
            rewards_t = torch.tensor(rews, dtype=torch.float32, device=self.device)
            n_board_t = torch.tensor(np.array(nb_s), dtype=torch.float32, device=self.device)
            n_cards_t = torch.tensor(np.array(nc_s), dtype=torch.float32, device=self.device)
            n_deck_t = torch.tensor(np.array(nd_s), dtype=torch.float32, device=self.device)
            dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
            dts_t = torch.tensor(dts, dtype=torch.float32, device=self.device)

            q_pred = self.net(board_t, cards_t, deck_t).gather(1, actions_t).squeeze(1)

            with torch.no_grad():
                # Double DQN: online net selects best next action, target net evaluates it
                next_actions = self.net(n_board_t, n_cards_t, n_deck_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(n_board_t, n_cards_t, n_deck_t).gather(1, next_actions).squeeze(1)
                gamma_dt = self.gamma ** dts_t
                q_target = rewards_t + (1.0 - dones_t) * gamma_dt * next_q

            loss = loss_fn(q_pred, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
            self.optimizer.step()
            losses.append(float(loss.item()))

        return float(np.mean(losses)) if losses else 0.0

    def train(
        self,
        n_iterations: int = 25,
        eval_interval: int = 5,
        log_file: str = "results/dqn_training_log.json",
    ) -> list[dict]:
        """Main Double DQN training loop."""
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        logs = []

        for iteration in range(1, n_iterations + 1):
            # Linearly decay epsilon across iterations
            frac = (iteration - 1) / max(1, n_iterations - 1)
            self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

            ep_stats = self.collect_games(self.games_per_iter)
            mean_loss = self.dqn_update()

            if iteration % self.target_sync_interval == 0:
                self.target_net.load_state_dict(self.net.state_dict())

            avg_reward = float(np.mean([s["ep_reward"] for s in ep_stats]))
            train_score = float(
                np.mean(
                    [
                        1.0 if s["terminal_reward"] > 0.05 else (0.0 if s["terminal_reward"] < -0.05 else 0.5)
                        for s in ep_stats
                    ]
                )
            )

            log_data = {
                "iteration": iteration,
                "epsilon": round(self.epsilon, 4),
                "replay_size": len(self.replay),
                "avg_reward": avg_reward,
                "train_accuracy_score": train_score,
                "q_loss": mean_loss,
                "timestamp": str(datetime.now()),
            }
            logs.append(log_data)

            print(
                f"[DQN] Iter {iteration}/{n_iterations} | "
                f"eps={self.epsilon:.2f} | "
                f"Avg Ep Reward: {avg_reward:+.3f} | "
                f"Train Score: {train_score*100:.1f}% | "
                f"Q-Loss: {mean_loss:.4f}",
                flush=True,
            )

            if iteration % eval_interval == 0 or iteration == n_iterations:
                ckpt_path = os.path.join(self.save_dir, f"ckpt_iter_{iteration}.pt")
                self.save_checkpoint(ckpt_path)
                self.save_checkpoint(os.path.join(self.save_dir, "best_model.pt"))

            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)

        return logs

    def evaluate_full_suite(self, n_games_per_opponent: int = 30, output_dir: str = "results") -> dict:
        """Run the standardized 3-baseline evaluation suite at depth 6."""
        if self.mover is None:
            self.mover = StockfishMover(depth=self.move_depth)
        ckpt_path = os.path.join(self.save_dir, "best_model.pt")
        self.save_checkpoint(ckpt_path)
        return evaluate_all_baselines(
            net=self.net,
            algo_name="dqn",
            member_name="Dhruv",
            mover=self.mover,
            device=self.device,
            n_games_per_opponent=n_games_per_opponent,
            max_rounds=self.max_rounds,
            checkpoint_path=ckpt_path,
            output_dir=output_dir,
        )

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "algo": "DQN",
                "model_state_dict": self.net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        if "target_state_dict" in checkpoint:
            self.target_net.load_state_dict(checkpoint["target_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
