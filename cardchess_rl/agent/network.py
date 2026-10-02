import torch
import torch.nn as nn

class PolicyValueNet(nn.Module):
    """Neural network for card selection.
    
    Architecture:
    - 3 Conv2D layers over board planes (16->32->64->64)
    - Flatten board features
    - Concatenate with card features (72) and deck state (15)
    - Shared FC layer (343 -> 256)
    - Policy head: FC(256->3) + Softmax -> pi(a|s)
    - Value head: FC(256->1) + Tanh -> V(s) in [-1, 1]
    """
    
    def __init__(self, card_dim: int = 72, deck_dim: int = 15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        board_flat = 64 * 8 * 8
        combined = board_flat + card_dim + deck_dim
        
        self.shared_fc = nn.Sequential(
            nn.Linear(combined, 256), nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, 3)
        self.value_head = nn.Sequential(
            nn.Linear(256, 1), nn.Tanh(),
        )
    
    def forward(self, board: torch.Tensor, cards: torch.Tensor, deck: torch.Tensor):
        """Forward pass."""
        x = self.conv(board)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, cards, deck], dim=1)
        x = self.shared_fc(x)
        
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
