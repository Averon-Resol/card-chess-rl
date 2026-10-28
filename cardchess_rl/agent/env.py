import random
from typing import Any, Callable, Dict, Optional, Tuple

import chess
import torch

from cardchess_rl.agent.encoding import encode_state
from cardchess_rl.engine.cards import ALL_CARDS, Card
from cardchess_rl.engine.game import CardChessGame
from cardchess_rl.search.evaluation import evaluate_for_side
from cardchess_rl.search.minimax import choose_move as minimax_choose_move
from cardchess_rl.search.stockfish_engine import StockfishMover, cp_to_potential


class CardChessEnv:
    """RL environment for card selection in Card-Chess (Revision 2: Reward v2 + Depth 6).

    The agent only makes card-selection decisions (action space = 3).
    Chess moves are handled by the move engine (Stockfish depth 6 by default,
    or Python minimax when `use_stockfish=False`).

    Literature-backed Reward v2:
      1. Win-probability potential (Suphx, Li et al. 2020; Lichess/Elo logistic mapping):
             Phi(s) = 2 * P_win(s) - 1 = tanh(cp * ln(10) / 800) in [-1, 1]
         where `cp` is the centipawn evaluation from the agent's perspective.
      2. SMDP potential-based reward shaping (Ng et al. 1999; Sutton et al. 1999):
             F(s, s', dt) = gamma^dt * Phi(s') - Phi(s),   with Phi(terminal) = 0.
         Telescoping across the agent's decision points guarantees policy invariance
         without double-counting intermediate opponent rounds.
      3. Margin-aware terminal adjudication (DouZero ADP, Zha et al. 2021):
             r_T = +1 (checkmate win), -1 (checkmate loss), 0 (true draw),
             or Phi(s_T) when the game reaches `max_rounds` without a mate/draw.
    """

    def __init__(
        self,
        agent_color: Optional[chess.Color] = chess.WHITE,
        move_depth: int = 6,
        gamma: float = 0.99,
        max_rounds: int = 60,
        use_stockfish: bool = True,
        mover: Optional[StockfishMover] = None,
        opponent_picker: Optional[Callable] = None,
    ):
        """Initialize the Card-Chess RL environment.

        Args:
            agent_color: chess.WHITE, chess.BLACK, or None to randomize each episode.
            move_depth: Search depth for chess moves (default 6 with Stockfish).
            gamma: Base per-round discount factor (default 0.99).
            max_rounds: Maximum rounds per episode before margin adjudication (default 60).
            use_stockfish: If True, use Stockfish C++ engine; else use Python minimax.
            mover: Optional shared StockfishMover instance to reuse across episodes.
            opponent_picker: Optional callback `(state, candidates) -> Card` for opponent
                             card selection. Defaults to uniform random.
        """
        self.fixed_agent_color = agent_color
        self.agent_color: chess.Color = agent_color if agent_color is not None else chess.WHITE
        self.move_depth = move_depth
        self.gamma = gamma
        self.max_rounds = max_rounds
        self.use_stockfish = use_stockfish
        self._owns_mover = False
        self.mover: Optional[StockfishMover] = mover
        if self.use_stockfish and self.mover is None:
            self.mover = StockfishMover(depth=self.move_depth)
            self._owns_mover = True
        self.opponent_picker = opponent_picker
        self.game: Optional[CardChessGame] = None

    def close(self) -> None:
        if self._owns_mover and self.mover is not None:
            self.mover.close()
            self.mover = None

    def _get_phi(self) -> float:
        """Compute bounded win-probability potential Phi(s) in [-1, 1]."""
        if self.game is None:
            return 0.0
        # Evaluate using fast static evaluation in centipawns, mapped via logistic tanh
        cp = evaluate_for_side(self.game.board, self.agent_color)
        return cp_to_potential(cp)

    def _select_move(self, board: chess.Board, moves: list[chess.Move]) -> Optional[chess.Move]:
        if not moves:
            return None
        if self.use_stockfish and self.mover is not None:
            return self.mover.choose_move(board, moves)
        return minimax_choose_move(board, moves, depth=self.move_depth)

    def _is_episode_over(self) -> bool:
        return self.game.is_game_over() or self.game.round_number > self.max_rounds

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the game and advance until the agent wins a coin toss to pick a card."""
        for _ in range(5):
            if self.fixed_agent_color is None:
                self.agent_color = random.choice([chess.WHITE, chess.BLACK])
            else:
                self.agent_color = self.fixed_agent_color
            self.game = CardChessGame()
            state_dict, done, _ = self._advance_to_agent_pick()
            if not done and state_dict is not None:
                return state_dict
        # Fallback: force agent as picker on fresh game so reset() always returns a valid state
        self.game = CardChessGame()
        _, candidates = self.game.start_round()
        self.game.picker = self.agent_color
        self.game.candidate_cards = candidates
        return encode_state(self.game.get_state())

    def step(self, action: int) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """Agent picks card at index `action` (0, 1, or 2).

        Returns:
            (next_state, total_reward, done, info)
            where `info` contains:
              - 'terminal_reward': float
              - 'shaping_reward': float
              - 'rounds_elapsed' / 'dt': int (number of rounds elapsed until next agent pick)
              - 'adjudicated': bool (True if terminated by max_rounds cap)
        """
        if self._is_episode_over():
            raise RuntimeError("Step called on terminal state")

        phi_before = self._get_phi()

        # 1. Agent picks a card
        st = self.game.get_state()
        candidates = st.candidate_cards
        action_idx = int(action) if 0 <= int(action) < len(candidates) else 0
        picked_card = candidates[action_idx]
        self.game.pick_card(picked_card, candidates)

        # 2. Both sides make a move under the card-modified rules
        self._play_move_pair()

        # 3. End round
        if not self.game.is_game_over():
            self.game.end_round()

        # 4. Advance through any subsequent opponent-picked rounds
        next_state, done, adv_info = self._advance_to_agent_pick()
        dt = adv_info["rounds_elapsed"]

        # 5. Compute single telescoping SMDP shaping reward (no double-counting!)
        phi_after = 0.0 if done else self._get_phi()
        shaping_reward = self._compute_shaping_reward(phi_before, phi_after, dt)

        # 6. Compute terminal reward (with margin adjudication if round cap hit)
        terminal_reward = 0.0
        adjudicated = False
        if done:
            if self.game.is_game_over() and self.game.result is not None:
                res = self.game.result
                terminal_reward = res if self.agent_color == chess.WHITE else -res
            else:
                # Reached max_rounds cap: adjudicate using win-probability potential
                adjudicated = True
                terminal_reward = self._get_phi()

        total_reward = terminal_reward + shaping_reward
        info = {
            "terminal_reward": float(terminal_reward),
            "shaping_reward": float(shaping_reward),
            "rounds_elapsed": int(dt),
            "dt": int(dt),
            "adjudicated": adjudicated,
        }

        if next_state is None:
            next_state = self._make_dummy_state()

        return next_state, float(total_reward), done, info

    def _play_move_pair(self) -> None:
        """Play one move for White and one move for Black under active card rules.

        If a restriction card leaves a player with zero card-legal moves while the
        board itself has standard legal moves, fall back to standard legal moves so
        chess turn parity is never broken (Revision 2 edge-case fix).
        """
        for side in (chess.WHITE, chess.BLACK):
            if self.game.is_game_over() or self.game.board.turn != side:
                continue
            moves = self.game.get_legal_moves(side)
            if not moves and list(self.game.board.legal_moves):
                moves = list(self.game.board.legal_moves)
            if moves:
                move = self._select_move(self.game.board, moves)
                if move is not None:
                    self.game.make_move(move)
                    self.game._update_terminal_status()

    def _advance_to_agent_pick(self) -> Tuple[Optional[Dict[str, torch.Tensor]], bool, Dict[str, Any]]:
        """Play through opponent-picked rounds until the agent wins the toss or the game ends."""
        rounds_elapsed = 0

        while not self._is_episode_over():
            picker, candidates = self.game.start_round()
            rounds_elapsed += 1

            if picker == self.agent_color:
                return (
                    encode_state(self.game.get_state()),
                    False,
                    {"rounds_elapsed": max(1, rounds_elapsed)},
                )

            # Opponent's turn to pick a card
            if self.opponent_picker is not None:
                picked_card = self.opponent_picker(self.game.get_state(), candidates)
            else:
                picked_card = random.choice(candidates)
            self.game.pick_card(picked_card, candidates)

            self._play_move_pair()

            if not self.game.is_game_over():
                self.game.end_round()

        return None, True, {"rounds_elapsed": max(1, rounds_elapsed)}

    def _compute_shaping_reward(self, old_phi: float, new_phi: float, rounds_elapsed: int) -> float:
        """SMDP potential-based shaping: gamma^dt * Phi(s') - Phi(s)."""
        return (self.gamma ** rounds_elapsed) * new_phi - old_phi

    def _make_dummy_state(self) -> Dict[str, torch.Tensor]:
        """Create a zeroed-out state dict for terminal states."""
        return {
            "board": torch.zeros(16, 8, 8, dtype=torch.float32),
            "cards": torch.zeros(72, dtype=torch.float32),
            "deck": torch.zeros(15, dtype=torch.float32),
        }
