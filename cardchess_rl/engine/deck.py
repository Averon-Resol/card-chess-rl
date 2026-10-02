import random
from typing import List, Optional
from cardchess_rl.engine.cards import Card, ALL_CARDS

class Deck:
    def __init__(self, cards: Optional[List[Card]] = None):
        """Initialize with all 15 cards."""
        if cards is None:
            self.cards = list(ALL_CARDS)
        else:
            self.cards = list(cards)
        self.discard_pile = []
        random.shuffle(self.cards)
    
    def draw(self, n: int = 3) -> List[Card]:
        """Draw n cards without replacement. If fewer remain, reshuffle first."""
        if len(self.cards) < n:
            self.reshuffle()
        drawn = []
        for _ in range(min(n, len(self.cards))):
            drawn.append(self.cards.pop())
        return drawn
    
    def discard(self, card: Card) -> None:
        """Remove the picked card from the deck permanently (until reshuffle)."""
        self.discard_pile.append(card)
    
    def return_cards(self, cards: List[Card]) -> None:
        """Return unpicked cards to the deck."""
        self.cards.extend(cards)
        random.shuffle(self.cards)
    
    def get_deck_state(self) -> List[bool]:
        """Return 15-dim binary vector of which cards are still available."""
        state = [False] * 15
        for card in self.cards:
            idx = int(card.id[1:]) - 1
            state[idx] = True
        return state
    
    def reshuffle(self) -> None:
        """Restore all 15 cards and shuffle."""
        self.cards.extend(self.discard_pile)
        self.discard_pile = []
        random.shuffle(self.cards)
    
    @property
    def remaining(self) -> int:
        """Number of cards left in deck."""
        return len(self.cards)
