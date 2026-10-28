import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    """Actor-Critic neural network for card selection (used by PPO and A2C).

    Architecture:
    - 3 Conv2D layers over board planes (16->32->64->64)
    - Flatten board features (64 * 8 * 8 = 4096)
    - Concatenate with card features (72) and deck state (15) -> 4183 dims
    - Shared FC layer (4183 -> 256)
    - Policy head: FC(256 -> 3) logits -> Softmax pi(a|s)
    - Value head: FC(256 -> 1) + Tanh -> V(s) in [-1, 1]
    """

    def __init__(self, card_dim: int = 72, deck_dim: int = 15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        board_flat = 64 * 8 * 8
        combined = board_flat + card_dim + deck_dim

        self.shared_fc = nn.Sequential(
            nn.Linear(combined, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, 3)
        self.value_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, board: torch.Tensor, cards: torch.Tensor, deck: torch.Tensor):
        """Forward pass returning (policy_logits [B, 3], value [B, 1])."""
        x = self.conv(board)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, cards, deck], dim=1)
        x = self.shared_fc(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


class QNet(nn.Module):
    """Deep Q-Network for card selection (used by Double DQN).

    Uses the exact same Conv2D + shared FC trunk as PolicyValueNet so that
    PPO, DQN, and A2C have identical state representation capacity (~1.13M params).
    Outputs Q(s, a) for a in {0, 1, 2}.
    """

    def __init__(self, card_dim: int = 72, deck_dim: int = 15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        board_flat = 64 * 8 * 8
        combined = board_flat + card_dim + deck_dim

        self.shared_fc = nn.Sequential(
            nn.Linear(combined, 256),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(256, 3)

    def forward(self, board: torch.Tensor, cards: torch.Tensor, deck: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values [B, 3]."""
        x = self.conv(board)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, cards, deck], dim=1)
        x = self.shared_fc(x)
        return self.q_head(x)
