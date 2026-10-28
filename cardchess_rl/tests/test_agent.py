import pytest
import torch
import chess
from cardchess_rl.engine.game import CardChessGame
from cardchess_rl.engine.state import GameState
from cardchess_rl.agent.encoding import encode_state, encode_board
from cardchess_rl.agent.network import PolicyValueNet, QNet
from cardchess_rl.agent.env import CardChessEnv
from cardchess_rl.search.stockfish_engine import StockfishMover, cp_to_potential


def test_encoding():
    game = CardChessGame()
    game.start_round()
    state = game.get_state()
    encoded = encode_state(state)

    assert "board" in encoded
    assert "cards" in encoded
    assert "deck" in encoded

    assert encoded["board"].shape == (16, 8, 8)
    assert encoded["cards"].shape == (72,)
    assert encoded["deck"].shape == (15,)


def test_network():
    net = PolicyValueNet()
    qnet = QNet()
    batch_size = 4
    board = torch.randn(batch_size, 16, 8, 8)
    cards = torch.randn(batch_size, 72)
    deck = torch.randn(batch_size, 15)

    policy, value = net(board, cards, deck)
    qvals = qnet(board, cards, deck)

    assert policy.shape == (batch_size, 3)
    assert value.shape == (batch_size, 1)
    assert qvals.shape == (batch_size, 3)


def test_reward_v2_potential():
    assert cp_to_potential(0.0) == pytest.approx(0.0)
    assert 0.0 < cp_to_potential(200.0) < 1.0
    assert -1.0 < cp_to_potential(-200.0) < 0.0
    assert cp_to_potential(10000.0) == pytest.approx(1.0, abs=1e-4)


def test_env_reset():
    env = CardChessEnv(agent_color=chess.WHITE, move_depth=1, use_stockfish=False)
    state = env.reset()

    assert state is not None
    assert "board" in state
    assert state["board"].shape == (16, 8, 8)


def test_env_step():
    env = CardChessEnv(agent_color=chess.WHITE, move_depth=1, use_stockfish=False)
    env.reset()

    next_state, reward, done, info = env.step(0)

    if not done:
        assert next_state is not None
        assert "board" in next_state
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "terminal_reward" in info
    assert "shaping_reward" in info
    assert "dt" in info and info["dt"] >= 1


def test_env_full_game():
    env = CardChessEnv(agent_color=chess.WHITE, move_depth=1, max_rounds=15, use_stockfish=False)
    state = env.reset()
    done = False

    steps = 0
    while not done and steps < 50:
        action = 0  # always pick first card
        state, reward, done, info = env.step(action)
        steps += 1

    assert done


def test_stockfish_depth_6_with_cards():
    mover = StockfishMover(depth=6)
    try:
        game = CardChessGame()
        _, cands = game.start_round()
        game.pick_card(cands[0], cands)
        moves = game.get_legal_moves(chess.WHITE)
        chosen = mover.choose_move(game.board, moves)
        assert chosen in moves
    finally:
        mover.close()
