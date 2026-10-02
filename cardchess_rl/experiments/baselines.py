import random
import torch
from cardchess_rl.search.evaluation import evaluate_for_side
from cardchess_rl.agent.encoding import encode_state
from cardchess_rl.engine.cards import Card
from cardchess_rl.engine.state import GameState

def random_picker(state: GameState, candidates: list[Card]) -> Card:
    """Pick a random card."""
    return random.choice(candidates)

def always_first_picker(state: GameState, candidates: list[Card]) -> Card:
    """Always pick the first card."""
    return candidates[0]

def greedy_picker(state: GameState, candidates: list[Card]) -> Card:
    """Pick the card that maximises one-ply evaluation.
    For each candidate card:
    1. Apply card (modify legal moves)
    2. For each legal move, evaluate the resulting position
    3. Pick the card whose best move gives the highest eval
    """
    best_card = candidates[0]
    best_eval = -float('inf')
    color = state.picker

    for card in candidates:
        board = state.board.copy()
        legal_moves = card.modify_legal_moves(board, color)
        
        if not legal_moves:
            # If no legal moves with this card, evaluate current board
            val = evaluate_for_side(board, color)
            if val > best_eval:
                best_eval = val
                best_card = card
            continue
            
        max_val = -float('inf')
        for move in legal_moves:
            board.push(move)
            val = evaluate_for_side(board, color)
            board.pop()
            if val > max_val:
                max_val = val
                
        if max_val > best_eval:
            best_eval = max_val
            best_card = card
            
    return best_card

def trained_picker(net: torch.nn.Module, device: str = 'cpu'):
    """Return a card picker function that uses the trained network.
    
    Usage: picker = trained_picker(loaded_net)
    Then pass picker as the card_picker argument.
    """
    def _pick(state: GameState, candidates: list[Card]) -> Card:
        with torch.no_grad():
            encoded = encode_state(state)
            board_t = encoded['board'].unsqueeze(0).to(device)
            cards_t = encoded['cards'].unsqueeze(0).to(device)
            deck_t = encoded['deck'].unsqueeze(0).to(device)
            
            policy_logits, _ = net(board_t, cards_t, deck_t)
            action_idx = torch.argmax(policy_logits, dim=1).item()
            if action_idx >= len(candidates):
                action_idx = 0
            return candidates[action_idx]
    return _pick
