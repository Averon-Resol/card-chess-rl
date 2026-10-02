import random
import chess
from typing import Tuple, List, Optional, Callable
from cardchess_rl.engine.cards import Card
from cardchess_rl.engine.deck import Deck
from cardchess_rl.engine.state import GameState

class CardChessGame:
    def __init__(self):
        """Initialize a new game with standard starting position."""
        self.board = chess.Board()
        self.deck = Deck()
        self.round_number = 1
        self.active_card: Optional[Card] = None
        self.game_over = False
        self.result: Optional[float] = None
        self.picker: Optional[chess.Color] = None
        self.candidate_cards: List[Card] = []
    
    def start_round(self) -> Tuple[chess.Color, List[Card]]:
        """Coin toss, draw 3 cards. Returns (picker_color, candidate_cards)."""
        self.picker = random.choice([chess.WHITE, chess.BLACK])
        self.candidate_cards = self.deck.draw(3)
        return self.picker, self.candidate_cards
    
    def pick_card(self, card: Card, candidates: List[Card]) -> None:
        """Picker chooses a card. Apply it, return unpicked to deck."""
        self.active_card = card
        unpicked = [c for c in candidates if c.id != card.id]
        self.deck.return_cards(unpicked)
        self.candidate_cards = []
    
    def get_legal_moves(self, color: chess.Color) -> List[chess.Move]:
        """Get legal moves for the given color, modified by active card."""
        if self.active_card:
            return list(self.active_card.modify_legal_moves(self.board, color))
        return list(self.board.legal_moves)
    
    def make_move(self, move: chess.Move) -> None:
        """Execute a move on the board."""
        self.board.push(move)
    
    def end_round(self) -> None:
        """End the current round, discard active card, check terminal."""
        if self.active_card:
            self.deck.discard(self.active_card)
            self.active_card = None
        
        self._update_terminal_status()
        self.round_number += 1
    
    def _update_terminal_status(self) -> None:
        outcome = self.board.outcome()
        if outcome is not None:
            self.game_over = True
            if outcome.winner == chess.WHITE:
                self.result = 1.0
            elif outcome.winner == chess.BLACK:
                self.result = -1.0
            else:
                self.result = 0.0
    
    def get_state(self) -> GameState:
        """Return current game state."""
        return GameState(
            board=self.board.copy(),
            active_card=self.active_card,
            round_number=self.round_number,
            picker=self.picker,
            candidate_cards=list(self.candidate_cards),
            deck_state=self.deck.get_deck_state(),
            is_terminal=self.game_over,
            result=self.result
        )
    
    def is_game_over(self) -> bool:
        """Check if game has ended."""
        return self.game_over
    
    def play_round(self, card_picker: Callable, move_maker_white: Callable, move_maker_black: Callable) -> GameState:
        """Play one complete round with provided decision functions."""
        if self.is_game_over():
            return self.get_state()
            
        picker, candidates = self.start_round()
        state = self.get_state()
        
        chosen_card = card_picker(state, candidates)
        self.pick_card(chosen_card, candidates)
        
        # White moves (if game not over and it's White's turn)
        if not self.is_game_over() and self.board.turn == chess.WHITE:
            moves = self.get_legal_moves(chess.WHITE)
            if moves:
                w_move = move_maker_white(self.get_state(), moves)
                self.make_move(w_move)
                self._update_terminal_status()
        
        # Black moves (if game not over and it's Black's turn)
        if not self.is_game_over() and self.board.turn == chess.BLACK:
            moves = self.get_legal_moves(chess.BLACK)
            if moves:
                b_move = move_maker_black(self.get_state(), moves)
                self.make_move(b_move)
                self._update_terminal_status()
                
        self.end_round()
        return self.get_state()
