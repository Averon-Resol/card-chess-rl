import time
import chess
from cardchess_rl.search.evaluation import evaluate, evaluate_for_side, evaluate_material_only

def test_starting_position():
    board = chess.Board()
    score = evaluate(board)
    # The starting position should be perfectly symmetric except for turn and mobility differences
    # Since White goes first, White has 20 legal moves. So mobility +40 for white.
    # Therefore it should be around 40, certainly between -100 and 100
    assert -100 <= score <= 100, f"Expected starting score to be close to 0, got {score}"

def test_removing_pawn():
    board = chess.Board()
    board.remove_piece_at(chess.E2) # Remove white pawn
    score = evaluate(board)
    assert score < -50, f"Expected negative score for white missing a pawn, got {score}"

def test_checkmate_detection():
    # Scholar's mate
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    board.push_san("Qxf7#")
    score = evaluate(board)
    assert score == 10000.0, f"Expected 10000.0 for white checkmating, got {score}"
    
def test_stalemate():
    # Known stalemate position
    board = chess.Board("8/8/8/8/8/7k/7p/7K w - - 0 1")
    score = evaluate(board)
    assert score == 0.0, f"Expected 0.0 for stalemate, got {score}"

def test_evaluate_for_side():
    board = chess.Board()
    board.remove_piece_at(chess.E2) # White missing pawn
    score_white = evaluate_for_side(board, chess.WHITE)
    score_black = evaluate_for_side(board, chess.BLACK)
    assert score_white == -score_black
    assert score_white < 0
    assert score_black > 0

def test_benchmark():
    board = chess.Board()
    start_time = time.time()
    for _ in range(10000):
        _ = evaluate(board)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\\nEvaluated 10000 positions in {elapsed:.4f} seconds")
    assert elapsed < 5.0, f"Evaluation too slow: {elapsed} seconds"
