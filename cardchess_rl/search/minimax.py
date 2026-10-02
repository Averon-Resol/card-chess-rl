import random
import math
import chess

from cardchess_rl.search.evaluation import evaluate

def order_moves(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """Order moves to improve alpha-beta pruning efficiency.
    
    Order: Captures, then checks, then quiet moves.
    """
    def move_score(move: chess.Move) -> int:
        score = 0
        if board.is_capture(move):
            score += 10
        if board.gives_check(move):
            score += 5
        return score

    return sorted(moves, key=move_score, reverse=True)

def minimax_search(
    board: chess.Board,
    depth: int = 3,
    legal_moves: list[chess.Move] | None = None,
    maximizing: bool = True,
    alpha: float = -math.inf,
    beta: float = math.inf,
) -> tuple[chess.Move | None, float]:
    """Find the best move using alpha-beta minimax search.
    
    Args:
        board: Current board position
        depth: Search depth (default 3)
        legal_moves: Override legal moves (for card-modified move sets).
                     If None, uses board.legal_moves.
        maximizing: True if maximizing player (White)
        alpha: Alpha value for pruning
        beta: Beta value for pruning
    
    Returns:
        (best_move, score) tuple
    """
    if depth == 0 or board.is_game_over():
        if board.is_checkmate():
            return None, -10000.0 if maximizing else 10000.0
        if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            return None, 0.0
        
        # Leaf node, evaluate position
        eval_score = evaluate(board)
        return None, eval_score

    moves = list(legal_moves) if legal_moves is not None else list(board.legal_moves)
    
    if not moves:
        return None, 0.0
        
    moves = order_moves(board, moves)

    best_move = None
    
    if maximizing:
        max_eval = -math.inf
        for move in moves:
            board.push(move)
            # Interior nodes use standard legal_moves (None)
            _, eval_score = minimax_search(board, depth - 1, None, False, alpha, beta)
            board.pop()
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return best_move, max_eval
    else:
        min_eval = math.inf
        for move in moves:
            board.push(move)
            _, eval_score = minimax_search(board, depth - 1, None, True, alpha, beta)
            board.pop()
            
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
                
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return best_move, min_eval

def choose_move(
    board: chess.Board,
    legal_moves: list[chess.Move] | None = None,
    depth: int = 3,
) -> chess.Move:
    """Choose the best move for the side to move.
    Uses alpha-beta minimax with the shared evaluation function.
    """
    maximizing = (board.turn == chess.WHITE)
    best_move, _ = minimax_search(
        board=board,
        depth=depth,
        legal_moves=legal_moves,
        maximizing=maximizing
    )
    if best_move is None:
        moves = list(legal_moves) if legal_moves is not None else list(board.legal_moves)
        return random.choice(moves) if moves else None
    return best_move

def choose_move_random(
    board: chess.Board,
    legal_moves: list[chess.Move] | None = None,
) -> chess.Move:
    """Choose a random legal move."""
    moves = list(legal_moves) if legal_moves is not None else list(board.legal_moves)
    return random.choice(moves) if moves else None
