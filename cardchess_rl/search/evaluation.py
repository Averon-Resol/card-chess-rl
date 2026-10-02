import chess

# Material values
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

# Piece-Square Tables (PST) mapped for python-chess squares (0=A1, 63=H8)
PST_PAWN = (
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
)

PST_KNIGHT = (
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
)

PST_BISHOP = (
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
)

PST_ROOK = (
      0,  0,  0,  0,  0,  0,  0,  0,
      5, 10, 10, 10, 10, 10, 10,  5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
     -5,  0,  0,  0,  0,  0,  0, -5,
      0,  0,  0,  5,  5,  0,  0,  0
)

PST_QUEEN = (
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
)

PST_KING_MIDDLEGAME = (
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
)

PST_KING_ENDGAME = (
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
)

PST = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN
}

def _mirror_square(sq: int) -> int:
    return sq ^ 56

def evaluate(board: chess.Board) -> float:
    """Evaluate a board position. Returns score from White's perspective.
    
    Positive = good for White, Negative = good for Black.
    Returns +/- 10000 for checkmate, 0 for stalemate/draw.
    """
    if board.is_checkmate():
        return -10000.0 if board.turn == chess.WHITE else 10000.0
    if board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0.0

    score = 0.0
    
    white_material = 0
    black_material = 0
    for piece_type in PIECE_VALUES:
        white_pieces = board.pieces(piece_type, chess.WHITE)
        black_pieces = board.pieces(piece_type, chess.BLACK)
        
        val = PIECE_VALUES[piece_type]
        white_count = len(white_pieces)
        black_count = len(black_pieces)
        
        white_material += white_count * val
        black_material += black_count * val
        
        score += (white_count - black_count) * val
        
        if piece_type in PST:
            pst = PST[piece_type]
            for sq in white_pieces:
                score += pst[sq]
            for sq in black_pieces:
                score -= pst[_mirror_square(sq)]
                
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 50
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 50
        
    total_material = white_material + black_material - len(board.pieces(chess.PAWN, chess.WHITE)) * 100 - len(board.pieces(chess.PAWN, chess.BLACK)) * 100
    phase = min(1.0, max(0.0, total_material / 6400.0))
    
    wk_sq = board.king(chess.WHITE)
    if wk_sq is not None:
        wk_mg = PST_KING_MIDDLEGAME[wk_sq]
        wk_eg = PST_KING_ENDGAME[wk_sq]
        score += wk_mg * phase + wk_eg * (1 - phase)
        
    bk_sq = board.king(chess.BLACK)
    if bk_sq is not None:
        bk_mg = PST_KING_MIDDLEGAME[_mirror_square(bk_sq)]
        bk_eg = PST_KING_ENDGAME[_mirror_square(bk_sq)]
        score -= bk_mg * phase + bk_eg * (1 - phase)

    mob = len(list(board.generate_legal_moves()))
    if board.turn == chess.WHITE:
        score += mob * 2
    else:
        score -= mob * 2
        
    return score

def evaluate_for_side(board: chess.Board, color: chess.Color) -> float:
    """Evaluate from the perspective of the given color.
    Positive = good for that color.
    """
    score = evaluate(board)
    return score if color == chess.WHITE else -score

def evaluate_material_only(board: chess.Board) -> float:
    """Fast material-only evaluation for testing."""
    score = 0.0
    for piece_type, val in PIECE_VALUES.items():
        score += (len(board.pieces(piece_type, chess.WHITE)) - len(board.pieces(piece_type, chess.BLACK))) * val
    return score
