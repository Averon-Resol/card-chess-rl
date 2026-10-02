import chess
import random
import torch
from typing import Tuple, Dict, Any, Optional

from cardchess_rl.engine.game import CardChessGame
from cardchess_rl.engine.cards import Card, ALL_CARDS
from cardchess_rl.search.evaluation import evaluate_for_side
from cardchess_rl.agent.encoding import encode_state

try:
    from cardchess_rl.search.minimax import choose_move
except ImportError:
    def choose_move(board: chess.Board, legal_moves=None, depth: int = 2) -> chess.Move:
        moves = list(legal_moves if legal_moves is not None else board.legal_moves)
        if moves:
            return random.choice(moves)
        return None

class CardChessEnv:
    """RL environment for card selection in Card-Chess.
    
    The agent only makes card-selection decisions (action space = 3).
    Chess moves are handled by the minimax engine.
    
    The environment steps through rounds until:
    1. The agent wins the coin toss and gets to pick a card (returns state)
    2. The game ends (returns terminal state)
    """
    
    def __init__(self, agent_color: chess.Color = chess.WHITE, 
                 move_depth: int = 2, gamma: float = 0.99):
        self.agent_color = agent_color
        self.move_depth = move_depth
        self.gamma = gamma
        self.game: Optional[CardChessGame] = None
        
    def _get_phi(self) -> float:
        """Get potential function Phi(s) for the current board state."""
        if self.game is None:
            return 0.0
        board = self.game.board
        return evaluate_for_side(board, self.agent_color) / 1000.0

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the game. Returns initial state when agent first gets to pick."""
        self.game = CardChessGame()
        # Advance to the first round where the agent is the picker
        state_dict, _, _, _ = self._advance_to_agent_pick(is_reset=True)
        if state_dict is None:
            # Game ended before agent got a turn (very unlikely)
            self.game = CardChessGame()
            state_dict, _, _, _ = self._advance_to_agent_pick(is_reset=True)
        return state_dict
        
    def step(self, action: int) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """Agent picks card at index `action` (0, 1, or 2).
        
        Internally:
        1. Apply chosen card
        2. Both sides make moves (minimax engine)
        3. End round
        4. Advance through rounds until agent picks again or game ends
        
        Returns: (next_state, total_reward, done, info)
        """
        if self.game.is_game_over():
            raise RuntimeError("Step called on terminal state")
            
        eval_before = self._get_phi()
        
        # 1. Pick the card
        st = self.game.get_state()
        candidates = st.candidate_cards
        picked_card = candidates[action]
        self.game.pick_card(picked_card, candidates)
        
        # 2. Both sides make moves under the card-modified rules
        self._play_move_pair()
            
        # 3. End round
        if not self.game.is_game_over():
            self.game.end_round()
            
        # 4. Advance through opponent rounds until agent's next turn or terminal
        next_state, shaping_acc, done, info = self._advance_to_agent_pick(is_reset=False)
        
        eval_after = self._get_phi() if not done else 0.0
        step_shaping = self._compute_shaping_reward(eval_before, eval_after, info['rounds_elapsed'])
        total_shaping = step_shaping + shaping_acc
        
        terminal_reward = 0.0
        if done:
            result = self.game.result
            if result is not None:
                if self.agent_color == chess.WHITE:
                    terminal_reward = result  # +1 white wins, -1 black wins, 0 draw
                else:
                    terminal_reward = -result  # flip for black agent
                    
        total_reward = terminal_reward + total_shaping
        info['terminal_reward'] = terminal_reward
        info['shaping_reward'] = total_shaping
        
        # If done, return a dummy state (zeros)
        if next_state is None:
            next_state = self._make_dummy_state()
        
        return next_state, total_reward, done, info
        
    def _play_move_pair(self) -> None:
        """Play one move for each side using the minimax engine."""
        # White's move
        if not self.game.is_game_over() and self.game.board.turn == chess.WHITE:
            moves = self.game.get_legal_moves(chess.WHITE)
            if moves:
                move = choose_move(self.game.board, moves, depth=self.move_depth)
                if move is not None:
                    self.game.make_move(move)
                    
        # Black's move
        if not self.game.is_game_over() and self.game.board.turn == chess.BLACK:
            moves = self.game.get_legal_moves(chess.BLACK)
            if moves:
                move = choose_move(self.game.board, moves, depth=self.move_depth)
                if move is not None:
                    self.game.make_move(move)
        
    def _advance_to_agent_pick(self, is_reset: bool = False) -> Tuple[Optional[Dict[str, torch.Tensor]], float, bool, Dict[str, Any]]:
        """Play through rounds where opponent picks until agent's turn or terminal.
        
        Returns: (state_dict_or_None, accumulated_shaping, done, info)
        """
        shaping_acc = 0.0
        rounds_elapsed = 0
        
        while not self.game.is_game_over():
            # Start a new round: coin toss + draw 3
            picker, candidates = self.game.start_round()
            rounds_elapsed += 1
            
            if picker == self.agent_color:
                # Agent's turn to pick — return state for the agent to decide
                return encode_state(self.game.get_state()), shaping_acc, False, {'rounds_elapsed': max(1, rounds_elapsed)}
            
            # Opponent's turn — pick randomly
            eval_before = self._get_phi()
            
            picked_card = random.choice(candidates)
            self.game.pick_card(picked_card, candidates)
            
            # Play moves
            self._play_move_pair()
            
            # End round
            if not self.game.is_game_over():
                self.game.end_round()
                
            eval_after = self._get_phi() if not self.game.is_game_over() else 0.0
            
            if not is_reset:
                shaping_acc += self._compute_shaping_reward(eval_before, eval_after, 1)
            
        return None, shaping_acc, True, {'rounds_elapsed': max(1, rounds_elapsed)}
        
    def _compute_shaping_reward(self, old_eval: float, new_eval: float, rounds_elapsed: int) -> float:
        """Potential-based shaping: gamma^dt * Phi(s') - Phi(s)"""
        return (self.gamma ** rounds_elapsed) * new_eval - old_eval
    
    def _make_dummy_state(self) -> Dict[str, torch.Tensor]:
        """Create a zeroed-out state dict for terminal states."""
        return {
            'board': torch.zeros(16, 8, 8, dtype=torch.float32),
            'cards': torch.zeros(72, dtype=torch.float32),
            'deck': torch.zeros(15, dtype=torch.float32),
        }
