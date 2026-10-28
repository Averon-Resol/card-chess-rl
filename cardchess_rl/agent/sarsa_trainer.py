import json
import os
import random
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


class SARSATrainer:
    """SARSA (on-policy TD control) with a neural Q-function (Owner: Dano).

    SARSA vs Q-learning (DQN): the TD target bootstraps from the action the
    behaviour policy ACTUALLY takes next, not from the greedy max:

        Q(s,a) <- Q(s,a) + alpha * [ r + gamma^dt * Q(s',a') - Q(s,a) ]

    where a' ~ epsilon-greedy(Q(s',.)). Because the policy being evaluated is
    the same one that generates the data, no replay buffer and no target network
    are used (classic on-policy SARSA). gamma^dt is the SMDP discount: dt is the
    number of rounds between two of the agent's decisions.

    Uses the same QNet trunk (~1.13M params) as DQN for a fair comparison.
    """

    def __init__(
        self,
        lr: float = 1e-4,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        epochs_per_iter: int = 2,
        batch_size: int = 64,
        games_per_iter: int = 16,
        move_depth: int = 6,
        max_rounds: int = 60,
        use_stockfish: bool = True,
        save_dir: str = "checkpoints/sarsa",
    ):
        self.lr = lr
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.epsilon = eps_start
        self.epochs_per_iter = epochs_per_iter
        self.batch_size = batch_size
        self.games_per_iter = games_per_iter
        self.move_depth = move_depth
        self.max_rounds = max_rounds
        self.use_stockfish = use_stockfish
        self.save_dir = save_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = QNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.mover: Optional[StockfishMover] = (
            StockfishMover(depth=self.move_depth) if self.use_stockfish else None
        )
        os.makedirs(self.save_dir, exist_ok=True)

    def close(self) -> None:
        if self.mover is not None:
            self.mover.close()
            self.mover = None

    def select_action(self, state: dict, greedy: bool = False) -> int:
        """Epsilon-greedy behaviour policy (the policy SARSA evaluates and improves)."""
        if (not greedy) and random.random() < self.epsilon:
            return random.randint(0, 2)
        board_t = state["board"].detach().clone().unsqueeze(0).to(self.device)
        cards_t = state["cards"].detach().clone().unsqueeze(0).to(self.device)
        deck_t = state["deck"].detach().clone().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(torch.argmax(self.net(board_t, cards_t, deck_t), dim=-1).item())

    def collect_games(self, n_games: int) -> tuple[list, list[dict]]:
        """Play episodes and return on-policy (s, a, r, s', a', done, dt) tuples."""
        self.net.eval()
        transitions, stats = [], []

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
            action = self.select_action(state)
            done, ep_reward, terminal_rew = False, 0.0, 0.0

            while not done:
                next_state, reward, done, info = env.step(action)
                # SARSA: a' is chosen NOW by the same epsilon-greedy policy and is
                # the action that will really be executed on the next step.
                next_action = 0 if done else self.select_action(next_state)
                transitions.append(
                    (
                        state["board"].numpy(), state["cards"].numpy(), state["deck"].numpy(),
                        int(action), float(reward),
                        next_state["board"].numpy(), next_state["cards"].numpy(), next_state["deck"].numpy(),
                        int(next_action), float(done), int(info.get("dt", 1)),
                    )
                )
                ep_reward += float(reward)
                if done:
                    terminal_rew = float(info.get("terminal_reward", 0.0))
                state, action = next_state, next_action

            stats.append({"ep_reward": ep_reward, "terminal_reward": terminal_rew})
        return transitions, stats

    def sarsa_update(self, transitions: list) -> float:
        """Semi-gradient SARSA update over this iteration's on-policy data."""
        if not transitions:
            return 0.0
        self.net.train()
        b_s, c_s, d_s, acts, rews, nb_s, nc_s, nd_s, nacts, dones, dts = zip(*transitions)
        board = torch.tensor(np.array(b_s), dtype=torch.float32, device=self.device)
        cards = torch.tensor(np.array(c_s), dtype=torch.float32, device=self.device)
        deck = torch.tensor(np.array(d_s), dtype=torch.float32, device=self.device)
        a_t = torch.tensor(acts, dtype=torch.int64, device=self.device).unsqueeze(1)
        r_t = torch.tensor(rews, dtype=torch.float32, device=self.device)
        n_board = torch.tensor(np.array(nb_s), dtype=torch.float32, device=self.device)
        n_cards = torch.tensor(np.array(nc_s), dtype=torch.float32, device=self.device)
        n_deck = torch.tensor(np.array(nd_s), dtype=torch.float32, device=self.device)
        na_t = torch.tensor(nacts, dtype=torch.int64, device=self.device).unsqueeze(1)
        done_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        dt_t = torch.tensor(dts, dtype=torch.float32, device=self.device)

        n = len(transitions)
        idx = np.arange(n)
        loss_fn = nn.SmoothL1Loss()
        losses = []
        for _ in range(self.epochs_per_iter):
            np.random.shuffle(idx)
            for s in range(0, n, self.batch_size):
                mb = torch.tensor(idx[s:s + self.batch_size], device=self.device)
                with torch.no_grad():
                    # TD target uses Q(s', a') for the action actually taken next.
                    q_next = self.net(n_board[mb], n_cards[mb], n_deck[mb]).gather(1, na_t[mb]).squeeze(1)
                    target = r_t[mb] + (1.0 - done_t[mb]) * (self.gamma ** dt_t[mb]) * q_next
                q_pred = self.net(board[mb], cards[mb], deck[mb]).gather(1, a_t[mb]).squeeze(1)
                loss = loss_fn(q_pred, target)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0

    def train(self, n_iterations: int = 25, eval_interval: int = 5,
              log_file: str = "results/sarsa_training_log.json") -> list[dict]:
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        logs = []
        for iteration in range(1, n_iterations + 1):
            frac = (iteration - 1) / max(1, n_iterations - 1)
            self.epsilon = self.eps_start + frac * (self.eps_end - self.eps_start)

            transitions, stats = self.collect_games(self.games_per_iter)
            loss = self.sarsa_update(transitions)

            avg_reward = float(np.mean([s["ep_reward"] for s in stats]))
            train_score = float(np.mean([
                1.0 if s["terminal_reward"] > 0.05 else (0.0 if s["terminal_reward"] < -0.05 else 0.5)
                for s in stats
            ]))
            logs.append({
                "iteration": iteration, "epsilon": round(self.epsilon, 4),
                "avg_reward": avg_reward, "train_accuracy_score": train_score,
                "td_loss": loss, "timestamp": str(datetime.now()),
            })
            print(
                f"[SARSA] Iter {iteration}/{n_iterations} | eps={self.epsilon:.2f} | "
                f"Avg Ep Reward: {avg_reward:+.3f} | Train Score: {train_score*100:.1f}% | TD-Loss: {loss:.4f}",
                flush=True,
            )
            if iteration % eval_interval == 0 or iteration == n_iterations:
                self.save_checkpoint(os.path.join(self.save_dir, f"ckpt_iter_{iteration}.pt"))
                self.save_checkpoint(os.path.join(self.save_dir, "best_model.pt"))
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
        return logs

    def evaluate_full_suite(self, n_games_per_opponent: int = 30, output_dir: str = "results") -> dict:
        if self.mover is None:
            self.mover = StockfishMover(depth=self.move_depth)
        ckpt_path = os.path.join(self.save_dir, "best_model.pt")
        self.save_checkpoint(ckpt_path)
        return evaluate_all_baselines(
            net=self.net, algo_name="sarsa", member_name="Dano", mover=self.mover,
            device=self.device, n_games_per_opponent=n_games_per_opponent,
            max_rounds=self.max_rounds, checkpoint_path=ckpt_path, output_dir=output_dir,
        )

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"algo": "SARSA", "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(), "epsilon": self.epsilon}, path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
