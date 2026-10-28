import json
import os
from datetime import datetime
from typing import Optional

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from cardchess_rl.agent.env import CardChessEnv
from cardchess_rl.agent.network import PolicyValueNet
from cardchess_rl.experiments.eval_suite import evaluate_all_baselines
from cardchess_rl.search.stockfish_engine import StockfishMover


class PPOTrainer:
    """Proximal Policy Optimization (PPO) trainer for Card-Chess (Revision 2: Depth 6 + SMDP GAE)."""

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        games_per_iter: int = 16,
        move_depth: int = 6,
        max_rounds: int = 60,
        use_stockfish: bool = True,
        save_dir: str = "checkpoints/ppo",
    ):
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.games_per_iter = games_per_iter
        self.move_depth = move_depth
        self.max_rounds = max_rounds
        self.use_stockfish = use_stockfish
        self.save_dir = save_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PolicyValueNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.mover: Optional[StockfishMover] = (
            StockfishMover(depth=self.move_depth) if self.use_stockfish else None
        )

        os.makedirs(self.save_dir, exist_ok=True)

    def close(self) -> None:
        if self.mover is not None:
            self.mover.close()
            self.mover = None

    def collect_games(self, n_games: int) -> list[dict]:
        """Play `n_games` episodes (alternating White and Black) and collect trajectories."""
        trajectories = []
        self.net.eval()

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

            states_board, states_cards, states_deck = [], [], []
            actions, rewards, log_probs, values, dones, dts = [], [], [], [], [], []
            terminal_rew = 0.0
            done = False

            while not done:
                board_tensor = state["board"].detach().clone().unsqueeze(0).to(self.device)
                cards_tensor = state["cards"].detach().clone().unsqueeze(0).to(self.device)
                deck_tensor = state["deck"].detach().clone().unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits, value = self.net(board_tensor, cards_tensor, deck_tensor)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                action_item = int(action.item())
                next_state, reward, done, info = env.step(action_item)

                states_board.append(state["board"].numpy())
                states_cards.append(state["cards"].numpy())
                states_deck.append(state["deck"].numpy())
                actions.append(action_item)
                rewards.append(reward)
                log_probs.append(float(log_prob.item()))
                values.append(float(value.item()))
                dones.append(done)
                dts.append(info.get("dt", 1))
                if done:
                    terminal_rew = float(info.get("terminal_reward", 0.0))

                state = next_state

            trajectories.append(
                {
                    "states_board": np.array(states_board, dtype=np.float32),
                    "states_cards": np.array(states_cards, dtype=np.float32),
                    "states_deck": np.array(states_deck, dtype=np.float32),
                    "actions": np.array(actions, dtype=np.int64),
                    "rewards": np.array(rewards, dtype=np.float32),
                    "log_probs": np.array(log_probs, dtype=np.float32),
                    "values": np.array(values, dtype=np.float32),
                    "dones": np.array(dones, dtype=np.float32),
                    "dts": np.array(dts, dtype=np.int32),
                    "terminal_reward": terminal_rew,
                }
            )

        return trajectories

    def compute_returns_and_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
         gamma: float,
        dts: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute SMDP-discounted returns and GAE advantages using gamma^dt."""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0.0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - float(dones[t])
            next_val = float(values[t + 1]) if t + 1 < len(rewards) else 0.0
            dt = int(dts[t]) if dts is not None else 1
            gamma_dt = gamma ** dt

            delta = rewards[t] + gamma_dt * next_val * next_non_terminal - values[t]
            last_gae_lam = delta + gamma_dt * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values
        return returns, advantages

    def ppo_update(self, batch: dict) -> dict:
        """Run PPO clipped surrogate updates over the collected batch."""
        self.net.train()
        board_tensor = torch.tensor(batch["states_board"], dtype=torch.float32).to(self.device)
        cards_tensor = torch.tensor(batch["states_cards"], dtype=torch.float32).to(self.device)
        deck_tensor = torch.tensor(batch["states_deck"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(batch["log_probs"], dtype=torch.float32).to(self.device)
        returns = torch.tensor(batch["returns"], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(batch["advantages"], dtype=torch.float32).to(self.device)

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = len(board_tensor)
        indices = np.arange(dataset_size)
        epoch_losses = {"policy": [], "value": [], "entropy": [], "total": []}

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                logits, values = self.net(
                    board_tensor[mb_idx], cards_tensor[mb_idx], deck_tensor[mb_idx]
                )
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[mb_idx])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])
                surr1 = ratio * advantages[mb_idx]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                    * advantages[mb_idx]
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values.squeeze(-1), returns[mb_idx])

                loss = (
                    policy_loss
                    + self.value_coeff * value_loss
                    - self.entropy_coeff * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()

                epoch_losses["policy"].append(policy_loss.item())
                epoch_losses["value"].append(value_loss.item())
                epoch_losses["entropy"].append(entropy.item())
                epoch_losses["total"].append(loss.item())

        return {k: float(np.mean(v)) for k, v in epoch_losses.items()}

    def train(
        self,
        n_iterations: int = 25,
        eval_interval: int = 5,
        log_file: str = "results/ppo_training_log.json",
    ) -> list[dict]:
        """Main PPO training loop."""
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        logs = []

        for iteration in range(1, n_iterations + 1):
            trajectories = self.collect_games(self.games_per_iter)

            all_board, all_cards, all_deck = [], [], []
            all_actions, all_old_log_probs, all_returns, all_advantages = [], [], [], []

            for traj in trajectories:
                returns, advantages = self.compute_returns_and_advantages(
                    traj["rewards"],
                    traj["values"],
                    traj["dones"],
                    self.gamma,
                    traj["dts"],
                )
                all_board.append(traj["states_board"])
                all_cards.append(traj["states_cards"])
                all_deck.append(traj["states_deck"])
                all_actions.append(traj["actions"])
                all_old_log_probs.append(traj["log_probs"])
                all_returns.append(returns)
                all_advantages.append(advantages)

            batch = {
                "states_board": np.concatenate(all_board),
                "states_cards": np.concatenate(all_cards),
                "states_deck": np.concatenate(all_deck),
                "actions": np.concatenate(all_actions),
                "log_probs": np.concatenate(all_old_log_probs),
                "returns": np.concatenate(all_returns),
                "advantages": np.concatenate(all_advantages),
            }

            losses = self.ppo_update(batch)
            avg_reward = float(np.mean([np.sum(t["rewards"]) for t in trajectories]))
            train_score = float(
                np.mean(
                    [
                        1.0 if t["terminal_reward"] > 0.05 else (0.0 if t["terminal_reward"] < -0.05 else 0.5)
                        for t in trajectories
                    ]
                )
            )

            log_data = {
                "iteration": iteration,
                "avg_reward": avg_reward,
                "train_accuracy_score": train_score,
                "losses": losses,
                "timestamp": str(datetime.now()),
            }
            logs.append(log_data)

            print(
                f"[PPO] Iter {iteration}/{n_iterations} | "
                f"Avg Ep Reward: {avg_reward:+.3f} | "
                f"Train Score: {train_score*100:.1f}% | "
                f"Loss: {losses['total']:.3f}",
                flush=True,
            )

            if iteration % eval_interval == 0 or iteration == n_iterations:
                ckpt_path = os.path.join(self.save_dir, f"ckpt_iter_{iteration}.pt")
                self.save_checkpoint(ckpt_path)
                self.save_checkpoint(os.path.join(self.save_dir, "best_model.pt"))

            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)

        return logs

    def evaluate_vs_random(self, n_games: int = 20) -> float:
        """Evaluate current policy against random card picker. Returns accuracy score (W + 0.5*D)/N."""
        score_sum = 0.0
        self.net.eval()
        for i in range(n_games):
            color = chess.WHITE if (i % 2 == 0) else chess.BLACK
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
            info = {}
            while not done:
                board_tensor = state["board"].detach().clone().unsqueeze(0).to(self.device)
                cards_tensor = state["cards"].detach().clone().unsqueeze(0).to(self.device)
                deck_tensor = state["deck"].detach().clone().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits, _ = self.net(board_tensor, cards_tensor, deck_tensor)
                    action = int(torch.argmax(logits, dim=-1).item())
                state, _, done, info = env.step(action)

            term = float(info.get("terminal_reward", 0.0))
            if term > 0.05:
                score_sum += 1.0
            elif abs(term) <= 0.05:
                score_sum += 0.5
        return score_sum / n_games if n_games > 0 else 0.0

    def evaluate_full_suite(self, n_games_per_opponent: int = 30, output_dir: str = "results") -> dict:
        """Run the standardized 3-baseline evaluation suite at depth 6."""
        if self.mover is None:
            self.mover = StockfishMover(depth=self.move_depth)
        ckpt_path = os.path.join(self.save_dir, "best_model.pt")
        self.save_checkpoint(ckpt_path)
        return evaluate_all_baselines(
            net=self.net,
            algo_name="ppo",
            member_name="Ihsal",
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
                "algo": "PPO",
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
