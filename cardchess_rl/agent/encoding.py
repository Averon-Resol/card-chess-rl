import torch
import chess
import numpy as np
from cardchess_rl.engine.state import GameState
from cardchess_rl.engine.cards import get_card_feature_vector

def encode_board(board: chess.Board) -> np.ndarray:
    """Encode board as 16 x 8 x 8 numpy array.
    
    Planes:
    0-5: White pieces (pawn, knight, bishop, rook, queen, king)
    6-11: Black pieces (pawn, knight, bishop, rook, queen, king)  
    12: Side to move (all 1s if White, all 0s if Black)
    13: White castling rights
    14: Black castling rights
    15: En passant square
    """
    encoded = np.zeros((16, 8, 8), dtype=np.float32)
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is not None:
            r = chess.square_rank(sq)
            c = chess.square_file(sq)
            plane = piece_to_plane[piece.piece_type]
            if piece.color == chess.BLACK:
                plane += 6
            encoded[plane, r, c] = 1.0
            
    if board.turn == chess.WHITE:
        encoded[12, :, :] = 1.0
        
    if board.has_kingside_castling_rights(chess.WHITE):
        encoded[13, :, :4] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        encoded[13, :, 4:] = 1.0
        
    if board.has_kingside_castling_rights(chess.BLACK):
        encoded[14, :, :4] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        encoded[14, :, 4:] = 1.0
        
    if board.ep_square is not None:
        r = chess.square_rank(board.ep_square)
        c = chess.square_file(board.ep_square)
        encoded[15, r, c] = 1.0
        
    return encoded

def encode_cards(candidate_cards: list) -> np.ndarray:
    """Encode 3 candidate cards as (3 * 24) = 72-dim vector."""
    if not candidate_cards:
        return np.zeros(72, dtype=np.float32)
    features = []
    for card in candidate_cards[:3]:
        features.extend(get_card_feature_vector(card))
    
    while len(features) < 72:
        features.append(0.0)
        
    return np.array(features, dtype=np.float32)

def encode_deck(deck_state: list[bool]) -> np.ndarray:
    """Encode deck state as 15-dim binary vector."""
    return np.array(deck_state, dtype=np.float32)

def encode_state(state: GameState) -> dict[str, torch.Tensor]:
    """Encode full state for the network.
    Returns dict with keys: 'board' (16,8,8), 'cards' (72,), 'deck' (15,)
    """
    board_encoded = encode_board(state.board)
    cards_encoded = encode_cards(state.candidate_cards)
    deck_encoded = encode_deck(state.deck_state)
    
    return {
        'board': torch.tensor(board_encoded, dtype=torch.float32),
        'cards': torch.tensor(cards_encoded, dtype=torch.float32),
        'deck': torch.tensor(deck_encoded, dtype=torch.float32)
    }
