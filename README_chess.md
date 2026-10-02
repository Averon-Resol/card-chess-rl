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
| **Module 1** | GUI (drag-and-drop, Lichess-style) | ⬜ Not Started | — | |
| **Module 2** | State encoding (board planes + cards + deck) | ✅ Done | — | 16×8×8 + 3×d_c + 15 |
| **Module 2** | Policy/Value network (Conv + MLP) | ✅ Done | — | ~3 conv layers, 2-head MLP |
| **Module 2** | Self-play training loop | ✅ Done | — | PPO with clipped surrogate |
| **Module 2** | Reward shaping (variable-interval Φ) | ✅ Done | — | γ^Δt adaptation |
| **Module 2** | Evaluation checkpoint pipeline | ✅ Done | — | vs frozen + random baseline |
| **Module 3** | Alpha-beta minimax search | ✅ Done | — | Depth-limited, card-aware |
| **Module 3** | Evaluation function Φ(s) | ✅ Done | — | Material + piece-square + mobility |
| **Module 3** | Experiment runner (win-rate benchmarks) | ✅ Done | — | RL vs random vs greedy vs always-first |
| **Cross-module** | CardChessEnv (RL environment wrapper) | ✅ Done | — | Gym-like API |

### Known Issues / Blockers

| Issue | Status | Impact |
|-------|--------|--------|
| 15 card definitions not yet provided | ✅ Resolved | Implemented 15 custom cards in engine/cards.py |
| Stockfish integration for benchmarking TBD | 🟡 Low priority | Only needed for post-training comparison |

### File Inventory

| File | Description |
|------|-------------|
| `README_chess.md` | This file — project spec + status tracker |
| `CardChessRL_LitReview.pdf` | Submitted deliverable: Goals 1–8 (background, lit review, gap analysis, research direction) |
| `CardChessRL_Goals9_14.docx` | Submitted deliverable: Goals 9–14 (RL formulation, diagrams, simulation plan, metrics, novelty, references) |
| `CardChessRL_Goals9_14.md` | Markdown source for Goals 9–14 (reference copy) |
| `cardchess_rl/engine/` | Module 1 core logic (`game.py`, `deck.py`, `cards.py`, `state.py`) |
| `cardchess_rl/search/` | Module 3 minimax engine (`minimax.py`, `evaluation.py`) |
| `cardchess_rl/agent/` | Module 2 RL agent (`network.py`, `encoding.py`, `env.py`, `training.py`) |
| `cardchess_rl/experiments/` | Experiment runner and baselines (`runner.py`, `baselines.py`) |
| `cardchess_rl/tests/` | Comprehensive pytest suite for all modules |
| `cardchess_rl/train.py` | Top-level script to launch PPO self-play training |

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
