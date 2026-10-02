from dataclasses import dataclass, field
import chess
from typing import List, Optional
from cardchess_rl.engine.cards import Card

@dataclass
class GameState:
    board: chess.Board
    active_card: Optional[Card]
    round_number: int
    picker: Optional[chess.Color]
    candidate_cards: List[Card]
    deck_state: List[bool]
    is_terminal: bool
    result: Optional[float]
    
    def copy(self) -> 'GameState':
        return GameState(
            board=self.board.copy(),
            active_card=self.active_card,
            round_number=self.round_number,
            picker=self.picker,
            candidate_cards=list(self.candidate_cards),
            deck_state=list(self.deck_state),
            is_terminal=self.is_terminal,
            result=self.result
        )
