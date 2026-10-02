import json
import time
import chess
from datetime import datetime
from cardchess_rl.engine.game import CardChessGame
from cardchess_rl.search.minimax import choose_move, choose_move_random
from cardchess_rl.experiments.baselines import random_picker, always_first_picker, greedy_picker

def play_match(
    picker_white,
    picker_black,
    move_depth: int = 1,
    max_rounds: int = 200,
) -> dict:
    """Play one complete game.
    Returns: {'result': 1.0/-1.0/0.0, 'rounds': int, 'moves': int}
    """
    game = CardChessGame()
    
    def move_maker_white(state, moves):
        if move_depth > 0:
            return choose_move(state.board, moves, move_depth)
        return choose_move_random(state.board, moves)
        
    def move_maker_black(state, moves):
        if move_depth > 0:
            return choose_move(state.board, moves, move_depth)
        return choose_move_random(state.board, moves)

    def match_card_picker(state, candidates):
        if state.picker == chess.WHITE:
            return picker_white(state, candidates)
        else:
            return picker_black(state, candidates)

    while not game.is_game_over() and game.round_number <= max_rounds:
        game.play_round(match_card_picker, move_maker_white, move_maker_black)

    if game.result is not None:
        result = game.result
    else:
        # Draw on max rounds
        result = 0.0
        
    # Calculate moves based on fullmove number (round to avoid errors if starting mid-game)
    moves = game.board.fullmove_number * 2 - (1 if game.board.turn == chess.WHITE else 0)
    
    return {
        'result': result,
        'rounds': game.round_number - 1,
        'moves': moves
    }

def run_experiment(
    picker_a,
    picker_b,
    n_games: int = 100,
    move_depth: int = 1,
    name: str = 'experiment',
) -> dict:
    """Run n_games games (half as white, half as black)."""
    start_time = time.time()
    wins_a = 0
    wins_b = 0
    draws = 0
    total_moves = 0
    
    for i in range(n_games):
        if i % 2 == 0:
            # A is White, B is Black
            res = play_match(picker_a, picker_b, move_depth=move_depth)
            result_val = res['result']
            if result_val == 1.0:
                wins_a += 1
            elif result_val == -1.0:
                wins_b += 1
            else:
                draws += 1
        else:
            # B is White, A is Black
            res = play_match(picker_b, picker_a, move_depth=move_depth)
            result_val = res['result']
            if result_val == 1.0:
                wins_b += 1
            elif result_val == -1.0:
                wins_a += 1
            else:
                draws += 1
                
        total_moves += res['moves']
        
    win_rate_a = (wins_a + 0.5 * draws) / max(1, n_games)
    avg_game_length = total_moves / max(1, n_games)
    
    return {
        'name': name,
        'n_games': n_games,
        'wins_a': wins_a,
        'wins_b': wins_b,
        'draws': draws,
        'win_rate_a': win_rate_a,
        'avg_game_length': avg_game_length,
        'time_seconds': time.time() - start_time
    }

def run_all_experiments(
    trained_picker=None,
    n_games: int = 50,
    move_depth: int = 1,
    output_file: str = 'experiment_results.json',
) -> list[dict]:
    """Run the standard experiment suite."""
    results = []
    
    experiments = [
        ("Random vs Random", random_picker, random_picker),
        ("Always-First vs Random", always_first_picker, random_picker),
        ("Greedy vs Random", greedy_picker, random_picker),
        ("Greedy vs Always-First", greedy_picker, always_first_picker),
    ]
    
    if trained_picker is not None:
        experiments.extend([
            ("Trained vs Random", trained_picker, random_picker),
            ("Trained vs Always-First", trained_picker, always_first_picker),
            ("Trained vs Greedy", trained_picker, greedy_picker),
        ])
        
    for name, p_a, p_b in experiments:
        print(f"Running {name}...")
        res = run_experiment(p_a, p_b, n_games=n_games, move_depth=move_depth, name=name)
        results.append(res)
        
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == '__main__':
    results = run_all_experiments(n_games=20, move_depth=1)
    for r in results:
        print(f"{r['name']}: Win rate A = {r['win_rate_a']:.1%}")
