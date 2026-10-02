import pytest
import chess
import time

from cardchess_rl.search.minimax import minimax_search, choose_move, choose_move_random

def test_mate_in_1():
    # Scholar's mate position before Qxf7#
    board = chess.Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    best_move, score = minimax_search(board, depth=2, maximizing=True)
    assert best_move == chess.Move.from_uci("f3f7")
    
def test_avoid_blunder():
    # White Q on d4, Black pawn on e5. White should not leave queen on d4.
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/3Q4/8/PPPPPPPP/RNB1KBNR w KQkq - 0 2")
    best_move, score = minimax_search(board, depth=2, maximizing=True)
    assert best_move is not None
    assert best_move == chess.Move.from_uci("d4e5")

def test_depth_0():
    board = chess.Board()
    best_move, score = minimax_search(board, depth=0, maximizing=True)
    assert best_move is None
    assert score == pytest.approx(0.0, abs=100)  # near-zero, small mobility bonus expected

def test_choose_move_returns_legal():
    board = chess.Board()
    legal_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    move = choose_move(board, legal_moves=legal_moves, depth=1)
    assert move in legal_moves

def test_choose_move_random():
    board = chess.Board()
    move = choose_move_random(board)
    assert move in list(board.legal_moves)
    
def test_benchmark_minimax():
    board = chess.Board()
    start = time.time()
    for _ in range(10):
        minimax_search(board, depth=3, maximizing=True)
    end = time.time()
    print(f"\nTime for 10 depth-3 searches: {end - start:.2f}s")
