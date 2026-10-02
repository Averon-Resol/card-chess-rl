import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import os
import json
from datetime import datetime
from cardchess_rl.agent.network import PolicyValueNet
from cardchess_rl.agent.env import CardChessEnv
import chess

class PPOTrainer:
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
        games_per_iter: int = 64,
        move_depth: int = 1,
        save_dir: str = 'checkpoints',
    ):
        """
        Initialize the PPO Trainer.
        """
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
        self.save_dir = save_dir
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.net = PolicyValueNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
        os.makedirs(self.save_dir, exist_ok=True)
    
    def collect_games(self, n_games: int) -> list[dict]:
        """Play n_games of self-play, collecting decision tuples.
        
        Returns list of trajectory dicts with keys:
        'states_board', 'states_cards', 'states_deck', 
        'actions', 'rewards', 'log_probs', 'values', 'dones'
        """
        trajectories = []
        
        for game_idx in range(n_games):
            env = CardChessEnv()
            state = env.reset()
            
            states_board = []
            states_cards = []
            states_deck = []
            actions = []
            rewards = []
            log_probs = []
            values = []
            dones = []
            
            done = False
            
            while not done:
                board_tensor = torch.tensor(state['board'], dtype=torch.float32).unsqueeze(0).to(self.device)
                cards_tensor = torch.tensor(state['cards'], dtype=torch.float32).unsqueeze(0).to(self.device)
                deck_tensor = torch.tensor(state['deck'], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits, value = self.net(board_tensor, cards_tensor, deck_tensor)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                
                action_item = action.item()
                next_state, reward, done, info = env.step(action_item)
                
                states_board.append(state['board'])
                states_cards.append(state['cards'])
                states_deck.append(state['deck'])
                actions.append(action_item)
                rewards.append(reward)
                log_probs.append(log_prob.item())
                values.append(value.item())
                dones.append(done)
                
                state = next_state
            
            trajectories.append({
                'states_board': np.array(states_board),
                'states_cards': np.array(states_cards),
                'states_deck': np.array(states_deck),
                'actions': np.array(actions),
                'rewards': np.array(rewards),
                'log_probs': np.array(log_probs),
                'values': np.array(values),
                'dones': np.array(dones)
            })
            
        return trajectories
    
    def compute_returns_and_advantages(
        self, rewards: np.ndarray, values: np.ndarray, 
        dones: np.ndarray, gamma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute discounted returns and GAE advantages."""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        returns = np.zeros_like(rewards, dtype=np.float32)
        
        last_gae_lam = 0
        last_value = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            
            # For the last step, we don't have next_value, use 0
            next_val = values[t+1] if t + 1 < len(rewards) else last_value
            delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
            
            advantages[t] = last_gae_lam = delta + gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            
        returns = advantages + values
        return returns, advantages
    
    def ppo_update(self, batch: dict) -> dict:
        """Run one PPO update epoch on a batch.
        Returns dict with loss components for logging.
        """
        board_tensor = torch.tensor(batch['states_board'], dtype=torch.float32).to(self.device)
        cards_tensor = torch.tensor(batch['states_cards'], dtype=torch.float32).to(self.device)
        deck_tensor = torch.tensor(batch['states_deck'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(batch['actions'], dtype=torch.int64).to(self.device)
        old_log_probs = torch.tensor(batch['log_probs'], dtype=torch.float32).to(self.device)
        returns = torch.tensor(batch['returns'], dtype=torch.float32).to(self.device)
        advantages = torch.tensor(batch['advantages'], dtype=torch.float32).to(self.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        dataset_size = len(board_tensor)
        indices = np.arange(dataset_size)
        
        epoch_losses = {'policy': [], 'value': [], 'entropy': [], 'total': []}
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                minibatch_idx = indices[start:end]
                
                mb_board = board_tensor[minibatch_idx]
                mb_cards = cards_tensor[minibatch_idx]
                mb_deck = deck_tensor[minibatch_idx]
                mb_actions = actions[minibatch_idx]
                mb_old_log_probs = old_log_probs[minibatch_idx]
                mb_returns = returns[minibatch_idx]
                mb_advantages = advantages[minibatch_idx]
                
                logits, values = self.net(mb_board, mb_cards, mb_deck)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(values.squeeze(-1), mb_returns)
                
                loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                epoch_losses['policy'].append(policy_loss.item())
                epoch_losses['value'].append(value_loss.item())
                epoch_losses['entropy'].append(entropy.item())
                epoch_losses['total'].append(loss.item())
                
        return {k: float(np.mean(v)) for k, v in epoch_losses.items()}
    
    def train(self, n_iterations: int = 100, eval_interval: int = 10, 
              log_file: str = 'training_log.json') -> None:
        """Main training loop."""
        
        logs = []
        
        for iteration in range(1, n_iterations + 1):
            print(f"Iteration {iteration}/{n_iterations}")
            
            trajectories = self.collect_games(self.games_per_iter)
            
            all_board, all_cards, all_deck, all_actions, all_old_log_probs, all_returns, all_advantages = [], [], [], [], [], [], []
            
            for traj in trajectories:
                returns, advantages = self.compute_returns_and_advantages(
                    traj['rewards'], traj['values'], traj['dones'], self.gamma
                )
                
                all_board.append(traj['states_board'])
                all_cards.append(traj['states_cards'])
                all_deck.append(traj['states_deck'])
                all_actions.append(traj['actions'])
                all_old_log_probs.append(traj['log_probs'])
                all_returns.append(returns)
                all_advantages.append(advantages)
                
            batch = {
                'states_board': np.concatenate(all_board),
                'states_cards': np.concatenate(all_cards),
                'states_deck': np.concatenate(all_deck),
                'actions': np.concatenate(all_actions),
                'log_probs': np.concatenate(all_old_log_probs),
                'returns': np.concatenate(all_returns),
                'advantages': np.concatenate(all_advantages)
            }
            
            losses = self.ppo_update(batch)
            
            avg_reward = np.mean([np.sum(t['rewards']) for t in trajectories])
            
            log_data = {
                'iteration': iteration,
                'avg_reward': float(avg_reward),
                'losses': losses,
                'timestamp': str(datetime.now())
            }
            logs.append(log_data)
            
            print(f"  Avg Reward: {avg_reward:.3f} | Total Loss: {losses['total']:.3f}")
            
            if iteration % eval_interval == 0:
                win_rate = self.evaluate_vs_random()
                print(f"  Evaluation Win Rate vs Random: {win_rate:.3f}")
                log_data['win_rate'] = win_rate
                
                self.save_checkpoint(os.path.join(self.save_dir, f'ckpt_iter_{iteration}.pt'))
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=4)
                
    def evaluate_vs_random(self, n_games: int = 50) -> float:
        """Evaluate current policy against random card picker.
        Returns win rate."""
        wins = 0
        for _ in range(n_games):
            env = CardChessEnv()
            state = env.reset()
            done = False
            
            while not done:
                board_tensor = torch.tensor(state['board'], dtype=torch.float32).unsqueeze(0).to(self.device)
                cards_tensor = torch.tensor(state['cards'], dtype=torch.float32).unsqueeze(0).to(self.device)
                deck_tensor = torch.tensor(state['deck'], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits, _ = self.net(board_tensor, cards_tensor, deck_tensor)
                    action = torch.argmax(logits, dim=-1).item()
                
                state, reward, done, _ = env.step(action)
                
            if reward > 0:
                wins += 1
                
        return wins / n_games if n_games > 0 else 0.0

    def save_checkpoint(self, path: str) -> None:
        """Save model weights and optimizer state."""
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model weights and optimizer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
