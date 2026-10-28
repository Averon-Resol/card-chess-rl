# Card-Chess RL — Project Specification

A chess variant where, once per round, a coin toss grants one player the choice of
1-of-3 randomly drawn "rule cards." The chosen card's effect applies to **both**
players for that round only. A human plays against a trained RL agent that has
learned, via self-play, which card to pick given the board state.

This document is the full spec for the three project modules. It assumes the
15 card definitions will be supplied separately and slotted into the interface
described in Module 1.

---

## Project Status Tracker

### Submission Progress

| # | Deadline Item | Status | Notes |
|---|---------------|--------|-------|
| 1 | Title and Team Details | ✅ Done | Submitted in `CardChessRL_LitReview.pdf` |
| 2 | Problem Background | ✅ Done | Submitted in `CardChessRL_LitReview.pdf` |
| 3 | Problem Statement | ✅ Done | Submitted in `CardChessRL_LitReview.pdf` |
| 4 | Why Reinforcement Learning? | ✅ Done | Submitted in `CardChessRL_LitReview.pdf` |
| 5 | Literature Review (6 papers) | ✅ Done | AlphaZero, Ng/Harada/Russell, PPO, TD-Gammon, Browne MCTS survey, DeepStack |
| 6 | Existing Systems / Products Study | ✅ Done | Fairy-Stockfish, lc0, Lichess/Chess.com |
| 7 | Gap Analysis | ✅ Done | No system combines RL card-drafting + rule-changing board game + shared search |
| 8 | Proposed Research Direction | ✅ Done | 3-module system: Game Engine, RL Card Agent, Move Engine |
| 9 | Initial RL Formulation | ✅ Done | Semi-MDP; state=board+cards+deck; action={1,2,3}; PPO objective. In `CardChessRL_Goals9_14.docx` |
| 10 | High-Level Solution Diagram | ✅ Done | System architecture, data flow, NN architecture diagrams. In `CardChessRL_Goals9_14.docx` |
| 11 | Dataset / Simulation Plan | ✅ Done | Pure self-play; ~50k–128k games; phased curriculum. In `CardChessRL_Goals9_14.docx` |
| 12 | Evaluation Metrics and Baseline | ✅ Done | Win rate vs random/always-first/greedy; ablations. In `CardChessRL_Goals9_14.docx` |
| 13 | Expected Novelty | ✅ Done | Variable-interval reward shaping; novel game variant. In `CardChessRL_Goals9_14.docx` |
| 14 | References | ✅ Done | 12 references. In `CardChessRL_Goals9_14.docx` |

### Implementation Progress

| Module | Component | Status | Owner | Notes |
|--------|-----------|--------|-------|-------|
| **Module 1** | python-chess board wrapper | ✅ Done | — | Fork/wrap `python-chess` |
| **Module 1** | Card effect framework (15 cards) | ✅ Done | — | Waiting for 15 card definitions |
| **Module 1** | Round/turn controller (coin toss, draw, apply) | ✅ Done | — | |
| **Module 1** | Deck management (draw-without-replacement, reshuffle) | ✅ Done | — | |
| **Module 1** | Game state serialization API | ✅ Done | — | |
| **Module 1** | GUI (drag-and-drop, Lichess-style) | ✅ Done | — | Flask + chessboard.js + Stockfish depth 6 |
| **Module 2** | State encoding (board planes + cards + deck) | ✅ Done | — | 16×8×8 + 3×d_c + 15 |
| **Module 2** | Policy/Value & Q networks (Conv + MLP) | ✅ Done | — | `PolicyValueNet` (PPO) + `QNet` (DQN/SARSA) |
| **Module 2** | 3-Algorithm Depth-6 Training Suite | ✅ Done | Ihsal / Dhruv / Dano | PPO (`train_ppo.py`), Double DQN (`train_dqn.py`), SARSA (`train_sarsa.py`) |
| **Module 2** | Literature-backed Reward v2 (variable-interval Φ) | ✅ Done | — | Win-prob potential + γ^Δt SMDP shaping + margin adjudication |
| **Module 2** | Standardized 3-Baseline Evaluation Suite | ✅ Done | — | `eval_suite.py` + `compare_results.py` |
| **Module 3** | Stockfish Depth-6 + Alpha-beta minimax search | ✅ Done | — | Card-aware `StockfishMover(depth=6)` + Python minimax |
| **Module 3** | Evaluation function Φ(s) | ✅ Done | — | Material + piece-square + mobility + logistic win-prob mapping |
| **Module 3** | Experiment runner (win-rate benchmarks) | ✅ Done | — | RL vs random vs greedy vs always-first |
| **Cross-module** | CardChessEnv (RL environment wrapper) | ✅ Done | — | Gym-like API with Reward v2 and Stockfish depth-6 mover |

### Known Issues / Blockers

| Issue | Status | Impact |
|-------|--------|--------|
| 15 card definitions not yet provided | ✅ Resolved | Implemented 15 custom cards in `engine/cards.py` |
| Stockfish integration for depth-6 move search | ✅ Resolved | Auto-downloaded in `search/stockfish_engine.py`, card-aware depth-6 search (~4 ms/move) |

### File Inventory

| File | Description |
|------|-------------|
| `README_chess.md` | This file — project spec + status tracker |
| `CardChessRL_LitReview.pdf` | Submitted deliverable: Goals 1–8 (background, lit review, gap analysis, research direction) |
| `CardChessRL_Goals9_14.docx` | Submitted deliverable: Goals 9–14 (RL formulation, diagrams, simulation plan, metrics, novelty, references) |
| `CardChessRL_Goals9_14.md` | Markdown source for Goals 9–14 (reference copy) |
| `MidReview_Prep_Guide.md` | Comprehensive mid-review study guide |
| `train_ppo.py` | **Member 1 (Ihsal):** Standalone script to train & evaluate **PPO** at depth 6 (`results/ppo_results.json`) |
| `train_dqn.py` | **Member 2 (Dhruv):** Standalone script to train & evaluate **Double DQN** at depth 6 (`results/dqn_results.json`) |
| `train_sarsa.py` | **Member 3 (Dano):** Standalone script to train & evaluate **SARSA** (on-policy TD control) at depth 6 (`results/sarsa_results.json`) |
| `compare_results.py` | Read-only leaderboard across PPO, DQN, SARSA; shows which model the app will use |
| `cardchess_rl/model_selector.py` | Reads scores A (PPO), B (DQN), C (SARSA) from `results/score_<algo>.txt` and picks the max. Set `FORCE_MODEL` there to override manually |
| `results/` and `models/` | **Committed on purpose.** `results/score_<algo>.txt` (appended accuracy per run), `results/<algo>_results.json`, `models/<algo>.pt` (trained weights) |
| `cardchess_rl/engine/` | Module 1 core logic (`game.py`, `deck.py`, `cards.py`, `state.py`) |
| `cardchess_rl/search/` | Module 3 search & eval (`stockfish_engine.py`, `minimax.py`, `evaluation.py`) |
| `cardchess_rl/agent/` | Module 2 RL agents (`network.py`, `encoding.py`, `env.py`, `training.py`, `dqn_trainer.py`, `sarsa_trainer.py`) |
| `cardchess_rl/experiments/` | Experiment runner, baselines, and depth-6 evaluation suite (`eval_suite.py`, `runner.py`, `baselines.py`) |
| `cardchess_rl/gui/` | Flask + chessboard.js interactive web demo (`app.py`, `templates/index.html`) |
| `cardchess_rl/tests/` | Comprehensive pytest suite (22 tests) for all modules |

---

## ⚠️ REVISION 2: Teacher Feedback After Mid-Review (2026-10-21) — READ THIS FIRST

> Future agents: this section **overrides** anything below that conflicts with it
> (especially §0 decision #5, §4.4 reward, §4.5 training algorithm).

### What the teacher said
1. **Reward is too weak.** A plain +1 / 0 / −1 terminal reward is likely detrimental (sparse, long games, most capped games end as a meaningless "draw"). Look at papers on similar problems and redesign the reward.
2. **Move-search depth 2–3 during training is far too shallow.** The minimax engine plays weak chess, so card effects aren't exploited and the learning signal is noise. Use **depth 6**.
3. **PPO may not be the right algorithm.** Train **3 algorithms** under identical conditions, compare them, and keep the best one for the final project.

### Agreed plan (decided with Ihsal, 2026-10-21)
| # | Decision |
|---|----------|
| R1 | **Depth 6 via Stockfish.** Our Python minimax measured 0.02 s/move at depth 2, 0.26 s at depth 3 and 1.7 s at depth 4 (~7× per ply), so depth 6 would take ≈80 s/move (≈2.5 h/game), which is infeasible. Stockfish (C++) is used as the **move engine** at depth 6 (~4 ms/move). Card rules are kept: Stockfish searches only the card-legal standard moves at the root. Card-added (non-standard) moves are pushed by us, and each resulting position is scored with Stockfish at depth 5 (= 6 plies total). Stockfish is auto-downloaded on first run. This **replaces §0 decision #5** (Stockfish was previously benchmark-only). |
| R2 | **Three algorithms, one per team member**, all with the same environment, reward, depth, training budget and evaluation: **PPO → Ihsal (`train_ppo.py`)**, **DQN (Double DQN) → Dhruv (`train_dqn.py`)**, **SARSA → Dano (`train_sarsa.py`)**. |
| R3 | **"Accuracy" metric = score** = (wins + ½·draws) / games, against the **random, always-first and greedy** card pickers, using the same fixed evaluation seeds for all three algorithms. The best overall score wins and is used in the final project/GUI (`compare_results.py`). |
| R4 | **New literature-based reward** (details in the "Reward v2" table below). |

### Reward v2 (literature-backed)
| Component | Formula | Source / motivation |
|---|---|---|
| Win-probability potential | Φ(s) = 2·P_win(s) − 1 = tanh(cp · ln10 / 800), where cp = centipawn eval from the agent's side | Suphx (Li et al., 2020) shapes rewards with a predictor of the *final outcome* instead of raw scores. The logistic cp→win-probability map is the standard Elo/Lichess conversion. This keeps Φ in [−1, 1], the same scale as the terminal reward. The old Φ = cp/1000 was unbounded (mate = ±10) and swamped the terminal signal. |
| SMDP shaping | F = γ^Δt · Φ(s′) − Φ(s), with Φ(terminal) = 0 | Ng, Harada & Russell (1999) potential-based shaping, extended to SMDPs/options with γ^τ (Sutton, Precup & Singh, 1999). Policy-invariant. **Bug fixed:** the old env also added per-opponent-round shaping, which double-counted. |
| Graded terminal | Checkmate win +1, loss −1. Real draw 0. **Round cap (60 rounds) is adjudicated as r_T = Φ(s_cap)**, not 0. | DouZero (Zha et al., 2021) showed margin-based (ADP) rewards beat pure win/loss (WP) on score efficiency. Capped games otherwise give a fake "draw" signal. Card-drafting RL (Vieira, Chaimowicz & Tavares, 2020) uses sparse ±1 and reports credit-assignment noise, which is the motivation for the dense shaping. |
| SMDP discount in learning targets | γ^Δt in GAE (PPO) and in the TD targets (DQN, SARSA) | Δt (rounds between agent decisions, geometric with mean 2) is now returned in `info["dt"]`. |
| Edge-case rule | If a restriction card leaves a side with **no** card-legal move, that side plays a normal legal move instead of skipping | Fixes the turn-order bug (B9 in the mid-review guide). |

### Team workflow (Revision 2)
1. `git pull`, then run **only your file**: `python train_ppo.py` (Ihsal), `python train_dqn.py` (Dhruv), `python train_sarsa.py` (Dano).
2. It prints the accuracy table and automatically writes `results/<algo>_results.json`, appends your score to `results/score_<algo>.txt` and saves `models/<algo>.pt`.
3. `git add results models`, commit, push. Each person touches different files, so there are no merge conflicts.
4. After everyone has pushed: `git pull`, `python compare_results.py`, then start the GUI. It uses the highest score automatically, or set `FORCE_MODEL` in `cardchess_rl/model_selector.py`.

### Status of Revision 2
| Item | Status |
|---|---|
| README updated with feedback (this section) | ✅ Done |
| Stockfish move engine + auto-download (`stockfish_engine.py`) | ✅ Done |
| Reward v2 + env fixes (`env.py`, `cards.py`, `game.py`) | ✅ Done |
| `train_ppo.py` / `train_dqn.py` / `train_sarsa.py` + shared `eval_suite.py` | ✅ Done |
| `compare_results.py` + GUI auto-promotion | ✅ Done |
| Each member runs ONLY their file, then commits `results/` and `models/` (see workflow below) | 🟡 Ready to run |
| Run `python compare_results.py` to crown the winner for the final demo | 🟡 Ready to run |

---

## 0. Confirmed Decisions

All open questions from the design discussion have been resolved. These are
settled decisions, not assumptions — build against them directly.

| # | Question | Decision |
|---|----------|----------|
| 1 | What is "one round"? | **One full move-pair** — coin toss → card chosen → Player A moves → Agent B moves → round ends → new toss. |
| 2 | Do the 15 cards deplete or refill? | **Drawn without replacement from a shared 15-card deck; reshuffle if the deck empties before the game ends.** |
| 3 | Do unpicked cards return to the deck? | **Yes — unpicked cards return to the deck; only the picked card is removed.** |
| 4 | Is a card always active, or is there a "no-op" option? | **A card is always active every round** (no explicit "skip" option). |
| 5 | What evaluation function backs the minimax engine and the RL reward-shaping potential Φ? | **See Section 5.1 below — Stockfish is available locally but is used only for post-training benchmarking, not for the training-loop evaluation function.** |
| 6 | UI: terminal, GUI, or local website? | **A GUI is required** — either a native GUI or a locally-run website, with drag-and-drop piece movement. Terminal-only is explicitly ruled out. See Section 4.4. |
| 7 | Self-play opponent pool: current weights only, or a checkpoint pool? | **Current-weights-vs-current-weights self-play**, with periodic evaluation against a frozen earlier checkpoint and a random-card baseline. |
| 8 | Framework choices | **Python, `python-chess` for board/move representation, PyTorch for the policy/value network.** |

---

## 1. Game Rules Summary

1. At the start of each round, a coin toss selects either the human or the agent as "picker" for that round.
2. Three cards are drawn (without replacement) from the 15-card deck and shown to both players.
3. The picker chooses one of the three. That card's effect is applied to **both** players for the duration of the round (see assumption #1 — currently: one full move-pair).
4. Each player then makes one move, under the modified rule if applicable, using standard chess legality otherwise.
5. The round ends; the picked card is discarded from the deck (unpicked cards return), and a new round begins with a fresh coin toss.
6. The game ends on checkmate, stalemate, draw, or another standard chess terminal condition.

---

## 2. System Architecture

```
                ┌─────────────────────────┐
                │   Module 1: Game Engine  │
                │   & Card System          │
                │  (board state, legality, │
                │   card effects, rounds)  │
                └────────────┬─────────────┘
                             │ board state + 3 candidate cards
                             ▼
                ┌─────────────────────────┐
                │  Module 2: RL Card-      │
                │  Selection Agent         │
                │  (policy/value network,  │
                │   self-play training)    │
                └────────────┬─────────────┘
                             │ chosen card
                             ▼
                ┌─────────────────────────┐
                │  Module 3: Move Engine,  │
                │  Evaluation & Experiments│
                │  (minimax/alpha-beta,    │
                │   eval function, testing)│
                └─────────────────────────┘
```

Module 1 owns the ground truth of game state. Module 2 only ever answers "which
of these 3 cards." Module 3 only ever answers "given the current (possibly
modified) rules, what's the best move," for both the human's legal-move
validation and the agent's actual piece movement.

---

## 3. Module 1 — Game Engine & Card System

**Owner responsibility:** everything about representing the game and applying
card effects; this module is the shared substrate the other two build on top of.

**Core components:**
- Fork or wrap `python-chess` board representation.
- **Card effect framework:** each of the 15 cards is implemented as a function
  (or small class) that patches the legal-move generator for the duration of
  one round — e.g. "queen additionally gains knight-move patterns." Design
  this as a pluggable interface so all 15 cards conform to the same shape:
  ```python
  class Card:
      id: str
      description: str
      def modify_legal_moves(self, board, color) -> set[Move]:
          ...  # returns additional/altered legal moves for this round
  ```
- **Round/turn controller:** coin toss → draw 3 cards from deck → present to
  picker → apply chosen card to both sides → hand off to Module 3 for the two
  moves → discard/reshuffle deck → repeat.
- **Deck management:** 15-card deck, without-replacement draws, reshuffle logic
  when exhausted (see assumption #2).
- **Legality safety net:** validate that every card's modification still
  produces a legal position (no leaving own king in check, no illegal capture
  of own pieces, etc.) — write unit tests per card once the 15 definitions
  arrive.
- **Game state serialization:** expose a clean state object (board + active
  card + round number + whose turn) that Modules 2 and 3 both consume, so
  neither needs to know about the other's internals.

**Deliverables:** a working game loop playable end-to-end by two humans (or
scripted dummy players) before RL or search components are plugged in.

---

## 4. Module 2 — RL Card-Selection Agent

**Owner responsibility:** the core RL contribution — learn, via self-play, which
of the 3 offered cards to pick given the board state.

### 4.1 State representation
- Board encoding: stack of 8×8 planes (one per piece type per color, standard
  AlphaZero-style encoding), plus side-to-move, plus any relevant metadata
  (round number, cards remaining in deck if that matters to strategy).
- Candidate-card encoding: small feature vector per card (which piece it
  affects, what the new move pattern is, any duration/magnitude parameter) —
  three such vectors per decision, one per offered card.

### 4.2 Network architecture
- A few convolutional layers over the board planes → flatten → concatenate
  with the 3 card feature vectors → small MLP → 3 output logits (one per
  candidate card) → softmax.
- Add a value head (shared trunk, separate output) predicting expected game
  outcome from the current state — used for the advantage baseline (see 4.4).
- This network is deliberately small: the action space is 3, not the ~4,672
  move-space of full chess RL, so a modest architecture (a handful of conv
  layers + one MLP head) should suffice. Start small and scale up only if
  training shows underfitting.

### 4.3 Action & decision points
- The agent only acts on rounds where the coin toss makes it the picker.
- On the human's picker-rounds, no agent action occurs (though the resulting
  board state still feeds into the agent's *next* decision).

### 4.4 Reward shaping & discounting
Let d₁, …, d_k be this player's card-pick decisions across one game.

**Terminal reward**, discounted back from game outcome:
```
R_terminal = +1 (win) / 0 (draw) / -1 (loss)
G_terminal(d_j) = γ^(k-j) · R_terminal        # γ ≈ 0.97–0.99
```

**Potential-based shaping**, using Module 3's evaluation function Φ(s):
```
r_shape(d_j) = γ · Φ(s_{j+1}) − Φ(s_j)
```
where s_j is the board state right before decision d_j and s_{j+1} is the state
right before the next decision. This is invariant to the optimal policy
(Ng, Harada & Russell 1999) and gives dense, immediate feedback on top of the
sparse terminal signal.

**Combined return:**
```
G(d_j) = G_terminal(d_j) + β · Σ_{j'=j}^{k} γ^(j'-j) · r_shape(d_j'),   β ≈ 0.1–0.3
```

**Baseline / advantage** (for variance reduction):
```
advantage(d_j) = G(d_j) − V(s_j)
```
Use `advantage` in the policy gradient loss rather than raw `G`.

**Sign-convention warning:** confirm Φ(s) is defined consistently (e.g.
positive = good for the side to move, or good for White — pick one and test
it), since a flipped sign will make the agent learn to sabotage itself and this
bug is hard to catch late.

### 4.5 Training algorithm
- REINFORCE with the baseline above, or PPO with a clipped surrogate objective
  (PPO is more forgiving of hyperparameter choices and recommended if time
  allows).
- Self-play: current policy plays against itself; Module 3's minimax engine
  executes the actual moves for both sides each round.
- Periodic evaluation checkpoints against: (a) a frozen earlier version of the
  policy, (b) a random-card-pick baseline — track win rate over time as the
  headline training-progress metric.

**Deliverables:** trained policy weights, a training-curve plot (win rate vs.
self-play iteration), and a frozen checkpoint used for the human-facing demo.

---

## 5. Module 3 — Move Engine, Evaluation & Experiments

**Owner responsibility:** actual move-making for both sides under whatever rule
is currently active, the evaluation function used both for move search and as
the RL reward-shaping potential Φ, and the experiments that validate the whole
system.

**Core components:**
- **Move search:** minimax with alpha-beta pruning, operating over whatever
  legal-move set Module 1 currently exposes (i.e., automatically respects
  active card effects without needing card-specific search logic).
- **Evaluation function:** material balance + positional heuristics (piece-
  square tables, king safety, mobility, etc.). Keep this fast — it's called
  both during search (many times per move) and during RL training (once per
  decision, for Φ).
- **Card-awareness check:** confirm the evaluation function doesn't need
  per-card special-casing — if a card is active, the modified legal-move set
  already reflects it, so search finds moves correctly without extra logic.
  Flag any card where this assumption breaks (e.g. cards with delayed or
  multi-round effects, if assumption #1 is revisited).
- **Experiments (this is what answers "does the RL part actually help"):**
  1. Trained RL card-picker vs. random-card-picker baseline — win rate.
  2. Trained RL card-picker vs. always-pick-first-option baseline — win rate.
  3. Training curve: win rate vs. self-play training iteration.
  4. (Optional, higher rigor) RL card-picker vs. a simple heuristic card-
     picker (e.g. greedy one-ply lookahead using the eval function directly,
     no learning) — this shows whether the *learning* is earning its keep
     versus a hand-coded heuristic doing the same job.

**Deliverables:** the move-search engine integrated with Module 1's legal-move
interface, the shared evaluation function used by both search and RL reward
shaping, and the final experiment results/plots for the report.

---

## 6. Cross-Module Interfaces

- **Module 1 → Module 2:** `GameState` object (board, active card, round
  number) + list of 3 `Card` objects when it's the agent's picker-round.
- **Module 2 → Module 1:** chosen `Card.id`.
- **Module 1 → Module 3:** `GameState` + current legal-move set (post any
  active card modification).
- **Module 3 → Module 1:** chosen `Move` to execute.
- **Module 3 → Module 2:** the evaluation function `Φ(state) -> float`, imported
  directly (not over an API) since both need it — Module 2 for reward shaping,
  Module 3 for search. Keep it as a single shared, well-tested function rather
  than two separate implementations.

---

## 7. Suggested Build Order

1. Module 1 builds the base game loop with *no* cards active — playable
   standard chess between two scripted/random players.
2. Module 3 plugs in minimax search against that base game loop.
3. Module 1 adds the card-effect framework and round/toss structure once the
   15 card definitions arrive; re-test legality per card.
4. Module 2 builds the state/action encoding and a training loop against
   Module 3's search engine, starting with just terminal reward (no shaping)
   to confirm the pipeline works end-to-end before adding reward shaping.
5. Layer in potential-based shaping and the value-head baseline once the basic
   loop trains without errors.
6. Run the Module 3 experiments to produce the win-rate comparisons for the
   final report.

---

## Changelog

| Date | Change | Files Affected |
|------|--------|----------------|
| 2026-09-28 | Initial project spec created | `README_chess.md` |
| 2026-09-28 | Submitted Goals 1–8 (lit review, background, gap analysis) | `CardChessRL_LitReview.pdf` |
| 2026-09-28 | Added Goals 9–14 (RL formulation, diagrams, simulation plan, evaluation metrics, novelty, references) | `CardChessRL_Goals9_14.docx`, `CardChessRL_Goals9_14.md` |
| 2026-09-28 | Updated README with project status tracker, file inventory, changelog, known issues | `README_chess.md` |
| 2026-10-03 | Implemented Modules 1, 2, 3 and Flask + chessboard.js GUI | `cardchess_rl/*` |
| 2026-10-09 | Added Mid-Review Prep Guide | `MidReview_Prep_Guide.md` |
| 2026-10-21 | **Revision 2 (Post-Mid-Review Teacher Feedback):** Implemented literature-backed Reward v2 (Suphx logistic win-probability potential + Ng/Sutton SMDP γ^Δt shaping + DouZero margin adjudication at round cap), integrated Stockfish C++ engine at depth 6 (`stockfish_engine.py`, auto-downloaded), added 3 standalone algorithm trainers (`train_ppo.py` for Ihsal, `train_dqn.py` for Dhruv, `train_a2c.py` for Dano), standardized 3-baseline evaluation (`eval_suite.py`), and leaderboard/auto-promotion (`compare_results.py`). | `README_chess.md`, `train_ppo.py`, `train_dqn.py`, `train_a2c.py`, `compare_results.py`, `cardchess_rl/*` |
| 2026-10-24 | **A2C replaced by SARSA** (Dano): on-policy TD control, no replay or target net, epsilon-greedy 1.0 to 0.05, gamma^dt TD target using the action actually taken next. Added `model_selector.py` (max of per-algorithm score files picks the GUI model, `FORCE_MODEL` override). Training now auto-writes `results/score_<algo>.txt` and committed `models/<algo>.pt`. | `train_sarsa.py`, `sarsa_trainer.py`, `model_selector.py`, `eval_suite.py`, `gui/app.py`, `compare_results.py` |


