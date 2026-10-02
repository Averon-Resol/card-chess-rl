import pytest
import chess
import random
from cardchess_rl.engine.cards import ALL_CARDS, _is_safe_move
from cardchess_rl.engine.deck import Deck
from cardchess_rl.engine.game import CardChessGame

def test_cards_valid_moves():
    board = chess.Board()
    for card in ALL_CARDS:
        moves = card.modify_legal_moves(board, chess.WHITE)
        for move in moves:
            # We don't want moves to leave king in check
            assert _is_safe_move(board, move)

def test_deck_management():
    deck = Deck()
    assert deck.remaining == 15
    drawn = deck.draw(3)
    assert len(drawn) == 3
    assert deck.remaining == 12
    deck.return_cards(drawn[:2])
    assert deck.remaining == 14
    deck.discard(drawn[2])
    assert deck.remaining == 14
    deck.reshuffle()
    assert deck.remaining == 15

def test_full_game_random():
    game = CardChessGame()
    
    def rand_card(state, cands):
        return random.choice(cands)
        
    def rand_move(state, moves):
        return random.choice(moves)
        
    rounds = 0
    while not game.is_game_over() and rounds < 50:
        game.play_round(rand_card, rand_move, rand_move)
        rounds += 1
        
    assert game.is_game_over() or rounds == 50
    if game.is_game_over():
        assert game.result in [1.0, 0.0, -1.0]

if __name__ == "__main__":
    pytest.main(["-v"])
