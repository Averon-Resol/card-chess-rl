from dataclasses import dataclass
from abc import ABC, abstractmethod
import chess
from typing import List, Set, Optional

def _is_safe_move(board: chess.Board, move: chess.Move) -> bool:
    """Manually test if a move leaves the king in check."""
    if move in board.legal_moves:
        return True
    
    piece = board.piece_at(move.from_square)
    target = board.piece_at(move.to_square)
    
    board.remove_piece_at(move.from_square)
    board.set_piece_at(move.to_square, piece)
    
    in_check = board.is_check()
    
    board.remove_piece_at(move.to_square)
    board.set_piece_at(move.from_square, piece)
    if target:
        board.set_piece_at(move.to_square, target)
        
    return not in_check

def _get_knight_moves(square: int) -> List[int]:
    moves = []
    file, rank = chess.square_file(square), chess.square_rank(square)
    offsets = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
    for df, dr in offsets:
        f, r = file + df, rank + dr
        if 0 <= f <= 7 and 0 <= r <= 7:
            moves.append(chess.square(f, r))
    return moves

def _get_extended_knight_moves(square: int) -> List[int]:
    moves = []
    file, rank = chess.square_file(square), chess.square_rank(square)
    offsets = [(1, 3), (3, 1), (-1, 3), (-3, 1), (1, -3), (3, -1), (-1, -3), (-3, -1)]
    for df, dr in offsets:
        f, r = file + df, rank + dr
        if 0 <= f <= 7 and 0 <= r <= 7:
            moves.append(chess.square(f, r))
    return moves

def _get_bishop_moves(board: chess.Board, square: int, color: chess.Color) -> List[int]:
    moves = []
    for df, dr in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        f, r = chess.square_file(square) + df, chess.square_rank(square) + dr
        while 0 <= f <= 7 and 0 <= r <= 7:
            sq = chess.square(f, r)
            p = board.piece_at(sq)
            if p:
                if p.color != color:
                    moves.append(sq)
                break
            moves.append(sq)
            f += df
            r += dr
    return moves

def _get_orthogonal_1_moves(square: int) -> List[int]:
    moves = []
    file, rank = chess.square_file(square), chess.square_rank(square)
    for df, dr in [(1,0), (-1,0), (0,1), (0,-1)]:
        f, r = file + df, rank + dr
        if 0 <= f <= 7 and 0 <= r <= 7:
            moves.append(chess.square(f, r))
    return moves

def _get_diagonal_1_moves(square: int) -> List[int]:
    moves = []
    file, rank = chess.square_file(square), chess.square_rank(square)
    for df, dr in [(1,1), (1,-1), (-1,1), (-1,-1)]:
        f, r = file + df, rank + dr
        if 0 <= f <= 7 and 0 <= r <= 7:
            moves.append(chess.square(f, r))
    return moves

def _get_king_moves(square: int) -> List[int]:
    return _get_orthogonal_1_moves(square) + _get_diagonal_1_moves(square)

class Card(ABC):
    """Base class for rule-modifying cards."""
    id: str
    name: str  
    description: str
    
    @abstractmethod
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        pass

# ADDITION CARDS
class QueenGainsKnightMoves(Card):
    id = "c1"
    name = "QueenGainsKnightMoves"
    description = "Queen can also move like a knight"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        moves = set(board.legal_moves)
        for sq in board.pieces(chess.QUEEN, color):
            for t_sq in _get_knight_moves(sq):
                target = board.piece_at(t_sq)
                if target is None or target.color != color:
                    m = chess.Move(sq, t_sq)
                    if _is_safe_move(board, m):
                        moves.add(m)
        return moves

class BishopGainsKnightMoves(Card):
    id = "c2"
    name = "BishopGainsKnightMoves"
    description = "Bishops can also move like a knight"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        moves = set(board.legal_moves)
        for sq in board.pieces(chess.BISHOP, color):
            for t_sq in _get_knight_moves(sq):
                target = board.piece_at(t_sq)
                if target is None or target.color != color:
                    m = chess.Move(sq, t_sq)
                    if _is_safe_move(board, m):
                        moves.add(m)
        return moves

class RookGainsDiagonal(Card):
    id = "c3"
    name = "RookGainsDiagonal"
    description = "Rooks can also move 1 square diagonally"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        moves = set(board.legal_moves)
        for sq in board.pieces(chess.ROOK, color):
            for t_sq in _get_diagonal_1_moves(sq):
                target = board.piece_at(t_sq)
                if target is None or target.color != color:
                    m = chess.Move(sq, t_sq)
                    if _is_safe_move(board, m):
                        moves.add(m)
        return moves

class KingGainsKnightMoves(Card):
    id = "c4"
    name = "KingGainsKnightMoves"
    description = "King can also move like a knight"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        moves = set(board.legal_moves)
        for sq in board.pieces(chess.KING, color):
            for t_sq in _get_knight_moves(sq):
                target = board.piece_at(t_sq)
                if target is None or target.color != color:
                    m = chess.Move(sq, t_sq)
                    if _is_safe_move(board, m):
                        moves.add(m)
        return moves

class PawnSidestep(Card):
    id = "c5"
    name = "PawnSidestep"
    description = "Pawns can also move 1 square sideways (left/right)"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        moves = set(board.legal_moves)
        for sq in board.pieces(chess.PAWN, color):
            file, rank = chess.square_file(sq), chess.square_rank(sq)
            for df in [-1, 1]:
                f = file + df
                if 0 <= f <= 7:
                    t_sq = chess.square(f, rank)
                    target = board.piece_at(t_sq)
                    if target is None or target.color != color:
                        m = chess.Move(sq, t_sq)
                        if _is_safe_move(board, m):
                            moves.add(m)
        return moves

class KnightGainsBishopMoves(Card):
    id = "c6"
    name = "KnightGainsBishopMoves"
    description = "Knights can also move like a bishop"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        moves = set(board.legal_moves)
        for sq in board.pieces(chess.KNIGHT, color):
            for t_sq in _get_bishop_moves(board, sq, color):
                m = chess.Move(sq, t_sq)
                if _is_safe_move(board, m):
                    moves.add(m)
        return moves

class BishopGainsOrthogonal(Card):
    id = "c7"
    name = "BishopGainsOrthogonal"
    description = "Bishops can also move 1 square orthogonally"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        moves = set(board.legal_moves)
        for sq in board.pieces(chess.BISHOP, color):
            for t_sq in _get_orthogonal_1_moves(sq):
                target = board.piece_at(t_sq)
                if target is None or target.color != color:
                    m = chess.Move(sq, t_sq)
                    if _is_safe_move(board, m):
                        moves.add(m)
        return moves

# RESTRICTION CARDS
class FrozenKnights(Card):
    id = "c8"
    name = "FrozenKnights"
    description = "Knights cannot move this round"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        return {m for m in board.legal_moves if board.piece_at(m.from_square).piece_type != chess.KNIGHT}

class FrozenBishops(Card):
    id = "c9"
    name = "FrozenBishops"
    description = "Bishops cannot move this round"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        return {m for m in board.legal_moves if board.piece_at(m.from_square).piece_type != chess.BISHOP}

class FrozenRooks(Card):
    id = "c10"
    name = "FrozenRooks"
    description = "Rooks cannot move this round"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        return {m for m in board.legal_moves if board.piece_at(m.from_square).piece_type != chess.ROOK}

class FrozenQueen(Card):
    id = "c11"
    name = "FrozenQueen"
    description = "Queen cannot move this round"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        return {m for m in board.legal_moves if board.piece_at(m.from_square).piece_type != chess.QUEEN}

class PawnsCannotCapture(Card):
    id = "c12"
    name = "PawnsCannotCapture"
    description = "Pawns can only push forward, no captures"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        moves = set()
        for m in board.legal_moves:
            p = board.piece_at(m.from_square)
            if p.piece_type == chess.PAWN and board.is_capture(m):
                continue
            moves.add(m)
        return moves

# SPECIAL CARDS
class AllPiecesKingMove(Card):
    id = "c13"
    name = "AllPiecesKingMove"
    description = "All pieces can additionally move 1 square in any direction"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        moves = set(board.legal_moves)
        for sq, piece in board.piece_map().items():
            if piece.color == color:
                for t_sq in _get_king_moves(sq):
                    target = board.piece_at(t_sq)
                    if target is None or target.color != color:
                        m = chess.Move(sq, t_sq)
                        if _is_safe_move(board, m):
                            moves.add(m)
        return moves

class PawnsCanMoveBackward(Card):
    id = "c14"
    name = "PawnsCanMoveBackward"
    description = "Pawns can retreat 1 square backward"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        moves = set(board.legal_moves)
        for sq in board.pieces(chess.PAWN, color):
            file, rank = chess.square_file(sq), chess.square_rank(sq)
            dr = -1 if color == chess.WHITE else 1
            r = rank + dr
            if 0 <= r <= 7:
                t_sq = chess.square(file, r)
                target = board.piece_at(t_sq)
                if target is None or target.color != color:
                    m = chess.Move(sq, t_sq)
                    if _is_safe_move(board, m):
                        moves.add(m)
        return moves

class KnightsExtendedRange(Card):
    id = "c15"
    name = "KnightsExtendedRange"
    description = "Knights can additionally make (3,1) jumps"
    def modify_legal_moves(self, board: chess.Board, color: chess.Color) -> set[chess.Move]:
        moves = set(board.legal_moves)
        for sq in board.pieces(chess.KNIGHT, color):
            for t_sq in _get_extended_knight_moves(sq):
                target = board.piece_at(t_sq)
                if target is None or target.color != color:
                    m = chess.Move(sq, t_sq)
                    if _is_safe_move(board, m):
                        moves.add(m)
        return moves

ALL_CARDS = [
    QueenGainsKnightMoves(), BishopGainsKnightMoves(), RookGainsDiagonal(), KingGainsKnightMoves(), 
    PawnSidestep(), KnightGainsBishopMoves(), BishopGainsOrthogonal(),
    FrozenKnights(), FrozenBishops(), FrozenRooks(), FrozenQueen(), PawnsCannotCapture(),
    AllPiecesKingMove(), PawnsCanMoveBackward(), KnightsExtendedRange()
]

def get_card_feature_vector(card: Card) -> list[float]:
    """Encode a card as a fixed-length feature vector for the RL agent."""
    # target_piece (6), effect_type (3), card_id (15)
    vec = [0.0] * 24
    
    # card_id
    idx = int(card.id[1:]) - 1
    vec[9 + idx] = 1.0
    
    # effect_type: add (0), restrict (1), special (2)
    if idx < 7:
        vec[6] = 1.0 # add
    elif idx < 12:
        vec[7] = 1.0 # restrict
    else:
        vec[8] = 1.0 # special
        
    # target piece: pawn(0), knight(1), bishop(2), rook(3), queen(4), king(5)
    piece_map = {
        "c1": 4, "c2": 2, "c3": 3, "c4": 5, "c5": 0, "c6": 1, "c7": 2,
        "c8": 1, "c9": 2, "c10": 3, "c11": 4, "c12": 0,
        "c13": -1, "c14": 0, "c15": 1
    }
    target = piece_map.get(card.id, -1)
    if target != -1:
        vec[target] = 1.0
    else:
        # For AllPiecesKingMove, could set all to 1 or leave all 0. We leave 0.
        pass
        
    return vec
