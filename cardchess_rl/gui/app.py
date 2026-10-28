import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from flask import Flask, render_template, request, jsonify
import chess
import torch

from cardchess_rl.engine.game import CardChessGame
from cardchess_rl.engine.cards import Card
from cardchess_rl.agent.network import PolicyValueNet, QNet
from cardchess_rl.search.minimax import choose_move
from cardchess_rl.search.stockfish_engine import StockfishMover
from cardchess_rl.experiments.baselines import greedy_picker, trained_picker

app = Flask(__name__)


def _load_best_card_picker():
    """Use the model chosen by cardchess_rl/model_selector.py; greedy heuristic if none trained yet."""
    from cardchess_rl.model_selector import best_model, read_scores

    scores = read_scores()
    print(f"[GUI] Scores  A(PPO)={scores['ppo']}  B(DQN)={scores['dqn']}  C(SARSA)={scores['sarsa']}")
    choice = best_model()
    if choice is not None:
        algo, path = choice
        try:
            ckpt = torch.load(path, map_location="cpu")
            net = PolicyValueNet() if algo == "ppo" else QNet()  # DQN and SARSA both use QNet
            net.load_state_dict(ckpt["model_state_dict"])
            print(f"[GUI] Using trained {algo.upper()} card-picker from {path}")
            return trained_picker(net, device="cpu")
        except Exception as exc:
            print(f"[GUI] Could not load {path}: {exc}")
    print("[GUI] No trained model found -> using Greedy-1Ply heuristic card-picker")
    return greedy_picker


try:
    _SHARED_MOVER = StockfishMover(depth=6)
except Exception:
    _SHARED_MOVER = None
_CARD_PICKER = _load_best_card_picker()


class WebGameController:
    def __init__(self):
        self.game = CardChessGame()
        self.human_color = chess.WHITE
        self.agent_color = chess.BLACK
        self.phase = "START_ROUND"
        self.moves_made_this_round = 0
        self.picker = None
        self.candidates = []
        self.step_state()

    def step_state(self):
        if self.game.is_game_over():
            self.phase = "GAME_OVER"
            return

        if self.phase == "START_ROUND":
            self.picker, self.candidates = self.game.start_round()
            self.phase = "PICK_CARD"
            self.step_state()
            
        elif self.phase == "PICK_CARD":
            if self.picker == self.agent_color:
                # Agent picks a card using trained RL model (or greedy fallback)
                card = _CARD_PICKER(self.game.get_state(), self.candidates)
                self.game.pick_card(card, self.candidates)
                self.phase = "MOVE"
                self.step_state()
                
        elif self.phase == "MOVE":
            if self.moves_made_this_round >= 2:
                self.game.end_round()
                self.moves_made_this_round = 0
                self.phase = "START_ROUND"
                self.step_state()
                return
                
            turn = self.game.board.turn
            if turn == self.agent_color:
                moves = self.game.get_legal_moves(self.agent_color)
                if moves:
                    if _SHARED_MOVER is not None:
                        move = _SHARED_MOVER.choose_move(self.game.board, moves)
                    else:
                        move = choose_move(self.game.board, moves, depth=3)
                    if move:
                        self.game.make_move(move)
                self.moves_made_this_round += 1
                self.step_state()

    def get_legal_moves_dict(self):
        if self.phase != "MOVE" or self.game.board.turn != self.human_color:
            return {}
        moves = self.game.get_legal_moves(self.human_color)
        moves_dict = {}
        for m in moves:
            from_sq = chess.square_name(m.from_square)
            to_sq = chess.square_name(m.to_square)
            if from_sq not in moves_dict:
                moves_dict[from_sq] = []
            moves_dict[from_sq].append(to_sq)
        return moves_dict

    def get_state_json(self):
        st = self.game.get_state()
        active_c = None
        if st.active_card:
            active_c = {"name": st.active_card.name, "desc": st.active_card.description}
            
        cands = []
        if self.phase == "PICK_CARD" and self.picker == self.human_color:
            cands = [{"id": c.id, "name": c.name, "desc": c.description} for c in self.candidates]
            
        return {
            "fen": self.game.board.fen(),
            "phase": self.phase,
            "picker": "Human" if self.picker == self.human_color else "Agent",
            "candidates": cands,
            "active_card": active_c,
            "legal_moves": self.get_legal_moves_dict(),
            "turn": "Human" if self.game.board.turn == self.human_color else "Agent",
            "game_over": self.game.is_game_over(),
            "result": self.game.result,
            "round": self.game.round_number
        }

controller = WebGameController()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def state():
    return jsonify(controller.get_state_json())

@app.route('/api/pick_card', methods=['POST'])
def pick_card():
    data = request.json
    card_id = data.get('card_id')
    
    if controller.phase == "PICK_CARD" and controller.picker == controller.human_color:
        selected = next((c for c in controller.candidates if c.id == card_id), None)
        if selected:
            controller.game.pick_card(selected, controller.candidates)
            controller.phase = "MOVE"
            controller.step_state()
            
    return jsonify(controller.get_state_json())

@app.route('/api/move', methods=['POST'])
def move():
    data = request.json
    from_sq = data.get('from')
    to_sq = data.get('to')
    promotion = data.get('promotion')
    
    if controller.phase == "MOVE" and controller.game.board.turn == controller.human_color:
        uci = from_sq + to_sq
        if promotion:
            uci += promotion
        
        move_obj = chess.Move.from_uci(uci)
        legal_moves = controller.game.get_legal_moves(controller.human_color)
        
        # Handle promotion logic if frontend didn't pass promotion but it's required
        if move_obj not in legal_moves:
            # try queen promotion
            prom_move = chess.Move.from_uci(uci + "q")
            if prom_move in legal_moves:
                move_obj = prom_move

        if move_obj in legal_moves:
            controller.game.make_move(move_obj)
            controller.moves_made_this_round += 1
            controller.step_state()
            
    return jsonify(controller.get_state_json())

@app.route('/api/reset', methods=['POST'])
def reset():
    global controller
    controller = WebGameController()
    return jsonify(controller.get_state_json())

if __name__ == '__main__':
    app.run(debug=True, port=5000)
