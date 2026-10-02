# AMRITA VISHWA VIDYAPEETHAM, AMRITAPURI
## B. Tech – Computer Science Engineering (Artificial Intelligence)
## 22AIE401 – Reinforcement Learning

**Team No:** 9
**Title:** CardChess-RL

| Name | Roll Number |
|------|-------------|
| Dhruv Nair | AM.SC.U4AIE23501 |
| Ihsal Riyas | AM.SC.U4AIE23037 |
| Dano Aby Alex | AM.SC.U4AIE23029 |

---

## 9. Initial RL Formulation: State, Action, Reward, Policy, Objective

### 9.1 MDP Formulation

The card-selection problem in Card-Chess RL is modelled as a **Semi-Markov Decision Process (Semi-MDP)** because the agent does not act at every game step — it only acts on rounds where the coin toss designates it as the picker. The intervals between successive decisions are variable and stochastic.

### 9.2 State Space (S)

The state $s_t$ observed by the agent at decision point $t$ is a composite of three components:

**Board Encoding (AlphaZero-style spatial planes):**
- 12 binary 8×8 planes — one per (piece-type, color) pair — encoding the current board position.
- 1 binary 8×8 plane indicating the side to move.
- 2 binary 8×8 planes for castling rights (kingside/queenside, per color).
- 1 binary 8×8 plane for en-passant square (if any).

Total board planes: **16 planes of size 8×8** (= 1,024 features).

**Candidate Card Encoding:**
Each of the 3 offered cards is encoded as a fixed-length feature vector:
- **Target piece type** (one-hot over 6 piece types): which piece the card modifies.
- **Effect type** (one-hot over effect categories: adds movement, restricts movement, grants immunity, etc.).
- **Magnitude / scope** (scalar): e.g. how many extra squares or whether the effect is global.

Each card vector has dimensionality $d_c$; the three cards are concatenated into a $3 \times d_c$ matrix.

**Deck State Encoding:**
- A 15-dimensional binary vector indicating which cards remain in the deck ($1$ = still available, $0$ = already used).
- This gives the agent partial information about what rule changes may appear in the future.

**Formal state:**
$$s_t = (\mathbf{B}_t,\ \mathbf{C}_t^{(1)},\ \mathbf{C}_t^{(2)},\ \mathbf{C}_t^{(3)},\ \mathbf{d}_t)$$

where $\mathbf{B}_t$ is the board tensor, $\mathbf{C}_t^{(i)}$ is the feature vector for the $i$-th candidate card, and $\mathbf{d}_t \in \{0,1\}^{15}$ is the deck state.

### 9.3 Action Space (A)

The action space is **discrete and fixed at 3**:
$$\mathcal{A} = \{1, 2, 3\}$$
corresponding to picking the 1st, 2nd, or 3rd offered card. This is deliberately small — the RL agent is responsible only for the card-selection decision, not for selecting chess moves (which is handled by the classical search engine in Module 3).

### 9.4 Transition Dynamics (T)

The transition $s_t \xrightarrow{a_t} s_{t+1}$ is governed by a sequence of events between two consecutive agent decisions:
1. The agent picks card $a_t$; the card's rule modification is applied.
2. Both players make one move each under the modified rules (moves are selected by the minimax engine, Module 3).
3. One or more rounds may pass where the opponent wins the coin toss (the agent does not act during these rounds).
4. When the agent next wins the coin toss, the resulting board position, new candidate cards, and updated deck form $s_{t+1}$.

Because of the coin toss and the opponent's own card picks, the transition is **stochastic** and the number of game-rounds between $s_t$ and $s_{t+1}$ is variable — hence the Semi-MDP framing.

### 9.5 Reward Function (R)

The reward signal combines a sparse terminal reward with a dense potential-based shaping term:

**Terminal reward:**
$$R_{\text{terminal}} = \begin{cases} +1 & \text{if the agent wins} \\ \ \ 0 & \text{if draw} \\ -1 & \text{if the agent loses} \end{cases}$$

**Potential-based shaping reward** (Ng, Harada & Russell, 1999):
$$r_{\text{shape}}(s_t, s_{t+1}) = \gamma^{\Delta t} \cdot \Phi(s_{t+1}) - \Phi(s_t)$$

where $\Phi(s)$ is the board evaluation function from Module 3 (material balance + positional heuristics), and $\Delta t$ is the number of game rounds elapsed between decision $t$ and decision $t{+}1$. The use of $\gamma^{\Delta t}$ rather than a fixed $\gamma$ is the project's adaptation to handle the variable-length gaps caused by the coin-toss mechanism.

**Combined return at decision $d_j$ (out of $k$ total decisions in a game):**
$$G(d_j) = \gamma^{k-j} \cdot R_{\text{terminal}} + \beta \sum_{j'=j}^{k} \gamma^{j'-j} \cdot r_{\text{shape}}(s_{j'}, s_{j'+1}), \quad \beta \approx 0.1\text{--}0.3$$

### 9.6 Policy (π)

The policy $\pi_\theta(a \mid s)$ is a parameterised stochastic policy implemented as a neural network:

$$\pi_\theta(a \mid s) = \text{softmax}\big(f_\theta(s)\big)_a, \quad a \in \{1, 2, 3\}$$

where $f_\theta(s)$ produces 3 logits. The network architecture (detailed in the Solution Diagram, Section 10) uses convolutional layers over the board planes, concatenated with the card and deck features, followed by an MLP head.

A **value head** $V_\phi(s)$ shares the same trunk and outputs a scalar estimate of the expected return from state $s$. This is used for advantage estimation and variance reduction.

### 9.7 Objective

The training objective is to maximise expected return under the policy:

$$J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{j=1}^{k} G(d_j)\right]$$

In practice, this is optimised using PPO's clipped surrogate objective (Schulman et al., 2017):

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\left(r_t(\theta) \cdot \hat{A}_t,\ \text{clip}\big(r_t(\theta),\ 1{-}\epsilon,\ 1{+}\epsilon\big) \cdot \hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio, $\hat{A}_t = G(d_t) - V_\phi(s_t)$ is the estimated advantage, and $\epsilon = 0.2$ is the clipping hyperparameter.

The value head is trained to minimise:
$$L^{V}(\phi) = \mathbb{E}_t\left[\big(V_\phi(s_t) - G(d_t)\big)^2\right]$$

The combined loss is:
$$L(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 \cdot L^{V}(\phi) - c_2 \cdot H[\pi_\theta]$$

where $H[\pi_\theta]$ is the entropy bonus (encouraging exploration) and $c_1 \approx 0.5$, $c_2 \approx 0.01$.

---

## 10. High-Level Solution Diagram

### 10.1 System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        CARD-CHESS RL SYSTEM                               │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    MODULE 1: GAME ENGINE                            │  │
│  │                                                                      │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │  │
│  │  │  python-chess │  │  Card Deck   │  │  Round Controller         │  │  │
│  │  │  Board State  │  │  Manager     │  │  (coin toss, draw 3,     │  │  │
│  │  │  & Legal Move │  │  (15 cards,  │  │   apply card, step,      │  │  │
│  │  │  Generator    │  │  draw/discard│  │   terminal check)        │  │  │
│  │  └──────┬───────┘  │  /reshuffle) │  └──────────┬────────────────┘  │  │
│  │         │           └──────┬───────┘             │                    │  │
│  │         └──────────────────┼─────────────────────┘                    │  │
│  │                            │                                          │  │
│  │                   GameState API                                       │  │
│  │          (board, active_card, round, legal_moves, deck_state)        │  │
│  └────────────────────────────┼──────────────────────────────────────────┘  │
│                               │                                            │
│              ┌────────────────┴────────────────┐                           │
│              ▼                                 ▼                           │
│  ┌─────────────────────────┐     ┌──────────────────────────────────────┐  │
│  │  MODULE 2: RL CARD-     │     │  MODULE 3: MOVE ENGINE &            │  │
│  │  SELECTION AGENT        │     │  EVALUATION                         │  │
│  │                         │     │                                      │  │
│  │  Input:                 │     │  Input:                              │  │
│  │   • Board tensor (16×   │     │   • Board state + modified           │  │
│  │     8×8 planes)         │     │     legal move set                   │  │
│  │   • 3 card vectors      │     │                                      │  │
│  │   • Deck state (15-dim) │     │  ┌───────────────────────┐           │  │
│  │                         │     │  │  Alpha-Beta Minimax   │           │  │
│  │  ┌───────────────────┐  │     │  │  Search (depth-       │           │  │
│  │  │ Conv Layers       │  │     │  │  limited, operates on │           │  │
│  │  │ (board planes)    │  │     │  │  card-modified legal  │           │  │
│  │  └────────┬──────────┘  │     │  │  move set)            │           │  │
│  │           │             │     │  └───────────┬───────────┘           │  │
│  │  ┌────────▼──────────┐  │     │              │                       │  │
│  │  │  Flatten +        │  │     │  ┌───────────▼───────────┐           │  │
│  │  │  Concatenate      │  │     │  │  Evaluation Function  │           │  │
│  │  │  (board + cards   │  │     │  │  Φ(s): material +     │           │  │
│  │  │   + deck)         │  │     │  │  piece-square tables  │◄──shared──│  │
│  │  └────────┬──────────┘  │     │  │  + king safety +      │           │  │
│  │           │             │     │  │  mobility              │           │  │
│  │  ┌────────▼──────────┐  │     │  └───────────────────────┘           │  │
│  │  │  MLP Head         │  │     │                                      │  │
│  │  │  ┌─────┐ ┌──────┐ │  │     │  Output: best Move                  │  │
│  │  │  │Pol. │ │Value │ │  │     └──────────────────────────────────────┘  │
│  │  │  │Head │ │Head  │ │  │                                               │
│  │  │  │(3)  │ │(1)   │ │  │                                               │
│  │  │  └──┬──┘ └──┬───┘ │  │                                               │
│  │  └─────┼───────┼─────┘  │                                               │
│  │        │       │        │                                               │
│  │   π(a|s)    V(s)       │                                               │
│  │   (softmax   (scalar   │                                               │
│  │    over 3)   estimate) │                                               │
│  │                         │                                               │
│  │  Output: chosen card ID │                                               │
│  └─────────────────────────┘                                               │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    TRAINING LOOP (Self-Play)                         │  │
│  │                                                                      │  │
│  │  current π_θ  vs  current π_θ  (both sides use Module 3 for moves) │  │
│  │           │                                                          │  │
│  │           ▼                                                          │  │
│  │  Collect trajectories {(s_t, a_t, G_t)} over N complete games       │  │
│  │           │                                                          │  │
│  │           ▼                                                          │  │
│  │  Compute advantages: Â_t = G(d_t) − V_ϕ(s_t)                       │  │
│  │           │                                                          │  │
│  │           ▼                                                          │  │
│  │  Update θ, ϕ via PPO clipped surrogate + value loss + entropy       │  │
│  │           │                                                          │  │
│  │           ▼                                                          │  │
│  │  Periodically evaluate against frozen checkpoint + random baseline  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Data Flow Per Round

```
  Coin Toss
      │
      ├── Agent wins toss ──► Draw 3 cards ──► Module 2: π_θ(a|s) ──► Card chosen
      │                                                                     │
      └── Opponent wins ──► Opponent picks card (random/heuristic) ─────────┘
                                                                            │
                                                                    Apply card effect
                                                                    (modify legal moves)
                                                                            │
                                                              ┌─────────────┴──────────────┐
                                                              ▼                            ▼
                                                       Module 3:                    Module 3:
                                                       Player A move               Player B move
                                                       (minimax search)            (minimax search)
                                                              │                            │
                                                              └─────────────┬──────────────┘
                                                                            │
                                                                    Round ends
                                                              (discard picked card,
                                                               return unpicked to deck)
                                                                            │
                                                                   Next round / 
                                                                   Terminal check
```

### 10.3 Neural Network Architecture Diagram

```
Input Layer
─────────────────────────────────────────────────────────
Board Planes (16 × 8 × 8)    Card Vectors (3 × d_c)    Deck (15)
       │                            │                      │
       ▼                            │                      │
  Conv2D(16→32, 3×3, ReLU)         │                      │
       │                            │                      │
       ▼                            │                      │
  Conv2D(32→64, 3×3, ReLU)         │                      │
       │                            │                      │
       ▼                            │                      │
  Conv2D(64→64, 3×3, ReLU)         │                      │
       │                            │                      │
       ▼                            │                      │
  Flatten (64 × 2 × 2 = 256)       │                      │
       │                            │                      │
       └────────────┬───────────────┘──────────────────────┘
                    │
                    ▼
            Concatenation
         (256 + 3×d_c + 15)
                    │
                    ▼
           FC(→256, ReLU)     ← shared trunk
                    │
           ┌────────┴────────┐
           ▼                 ▼
    Policy Head         Value Head
    FC(256→3)           FC(256→1)
    Softmax             Tanh
       │                   │
    π(a|s)              V(s) ∈ [−1, +1]
```

---

## 11. Dataset / Simulation Plan

### 11.1 No External Dataset — Pure Self-Play Generation

Card-Chess RL is an entirely novel game variant; no human-play dataset exists. All training data is generated through **self-play simulation**, following the AlphaZero paradigm. This is consistent with the rationale stated in Section 3 (Why Reinforcement Learning?): the absence of labelled data is a core motivation for the RL approach.

### 11.2 Self-Play Data Generation Pipeline

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Games per training iteration | 256 | Balances sample diversity with compute time |
| Total training iterations | ~200–500 | Scaled to convergence of win-rate curve |
| Total games (projected) | ~50,000–128,000 | Sufficient for a 3-action policy to converge |
| Max rounds per game | 200 (100 full move-pairs) | Prevents infinite games; draws declared if reached |
| Discount factor γ | 0.97–0.99 | Standard for episodic games with ~30–60 decisions |
| Shaping weight β | 0.1–0.3 | Tuned to prevent shaping from dominating terminal signal |
| PPO clip ε | 0.2 | Standard PPO hyperparameter |
| Learning rate | 1×10⁻⁴ to 3×10⁻⁴ | Adam optimiser, with linear decay |
| Mini-batch size | 64 | Per PPO update epoch |
| PPO epochs per iteration | 4 | Standard PPO practice |

### 11.3 Simulation Environment

The simulation environment is the **Module 1 Game Engine** itself, wrapped as a standard RL environment with the following interface:

```python
class CardChessEnv:
    def reset(self) -> State:
        """Reset board to starting position, reshuffle deck."""
    
    def step(self, card_choice: int) -> tuple[State, float, bool, dict]:
        """
        Agent picks a card (0, 1, or 2).
        Internally: apply card, Module 3 plays both moves,
        advance rounds until agent's next pick or terminal.
        Returns: (next_state, shaped_reward, done, info)
        """
    
    def get_legal_cards(self) -> list[Card]:
        """Return the 3 currently offered cards."""
```

### 11.4 Data Collected Per Game

Each complete self-play game produces a trajectory of decision tuples:

$$\tau = \{(s_1, a_1, r_1, s_2), (s_2, a_2, r_2, s_3), \ldots, (s_k, a_k, R_{\text{terminal}}, s_{\text{term}})\}$$

Across all games in one iteration, these tuples are aggregated into a replay buffer for PPO updates. Since PPO is an on-policy algorithm, the buffer is cleared after each training iteration.

### 11.5 Curriculum / Phased Training Plan

| Phase | Description | Purpose |
|-------|-------------|---------|
| **Phase 0** | Train Module 3's minimax engine on standard chess (no cards) | Validate search correctness |
| **Phase 1** | Self-play with terminal reward only ($\beta = 0$) | Confirm the training pipeline works end-to-end |
| **Phase 2** | Add potential-based shaping ($\beta = 0.1$) | Accelerate learning with dense feedback |
| **Phase 3** | Tune $\beta$, $\gamma$, learning rate via grid search | Optimise final performance |
| **Phase 4** | Freeze best checkpoint for evaluation experiments | Produce final results |

---

## 12. Evaluation Metrics and Baseline

### 12.1 Primary Metric

**Win Rate (%)** of the trained RL card-selection policy against baseline card-selection strategies, holding the move engine (Module 3) constant across all conditions. This isolates the contribution of the learned card-selection policy from the quality of move play.

Each evaluation consists of **200 games** (100 as White, 100 as Black) to account for first-move advantage. Results are reported with 95% confidence intervals.

### 12.2 Baseline Strategies

| Baseline | Description | Purpose |
|----------|-------------|---------|
| **Random Picker** | Selects one of the 3 offered cards uniformly at random | Lower-bound baseline; any competent policy should beat this |
| **Always-First Picker** | Always picks the first card presented | Tests whether the agent is learning position-dependent preferences vs. a fixed bias |
| **Greedy Heuristic Picker** | For each of the 3 candidate cards, simulates one ply with Module 3's evaluation function $\Phi(s)$ and picks the card yielding the highest $\Phi(s')$ | Upper-bound heuristic baseline; shows whether RL's multi-step reasoning outperforms myopic optimisation |

### 12.3 Secondary Metrics

| Metric | Definition | Purpose |
|--------|------------|---------|
| **Training Win Rate Curve** | Win rate of current policy vs. frozen checkpoint from iteration $N{-}50$, plotted over training iterations | Shows learning progress and convergence |
| **Average Game Length** | Mean number of rounds per game | Detects degenerate strategies (e.g. agent stalling) |
| **Card Selection Entropy** | $H[\pi_\theta] = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$, averaged over evaluation games | Measures whether the agent develops strong card preferences vs. near-uniform selection |
| **Shaping Reward Correlation** | Pearson correlation between cumulative shaping reward and game outcome | Validates that $\Phi$ is a useful proxy for win probability |
| **Elo Rating** (optional) | Relative Elo computed from pairwise win rates among all baselines + trained agent | Provides a single-number ranking if multiple agent checkpoints are compared |

### 12.4 Statistical Testing

Win-rate differences between the trained agent and each baseline will be tested for significance using:
- **Binomial proportion test** (for win/loss outcomes).
- **Bootstrapped confidence intervals** (1,000 resamples) for the win-rate difference.

A result is considered significant at $p < 0.05$.

### 12.5 Ablation Studies

| Ablation | Modification | Tests |
|----------|-------------|-------|
| **No shaping** | Set $\beta = 0$; train with terminal reward only | Whether potential-based shaping accelerates learning |
| **No deck encoding** | Remove $\mathbf{d}_t$ from the state | Whether knowledge of remaining deck improves strategy |
| **No value baseline** | Use raw $G(d_t)$ instead of advantage $\hat{A}_t$ in policy gradient | Whether the value head reduces variance meaningfully |

---

## 13. Expected Novelty

### 13.1 Core Novel Contribution

**The integration of per-round rule modification into a self-play RL training loop for a board game.**

No prior work combines all three of the following in a single system:
1. An RL agent that selects temporary rule modifications from a stochastic menu (card drafting).
2. A classical search engine whose legal-move set changes every round based on the selected rule.
3. Self-play training where the agent learns both to exploit rule changes for itself and to anticipate rule changes by the opponent.

Each of these elements exists independently in the literature (AlphaZero for self-play board game RL; CCG drafting for card selection; Fairy-Stockfish for multi-variant search), but their combination is novel.

### 13.2 Technical Novelty: Variable-Interval Potential-Based Reward Shaping

The classical potential-based shaping formula of Ng, Harada & Russell (1999) assumes the agent acts at every timestep:

$$r_{\text{shape}} = \gamma \cdot \Phi(s') - \Phi(s)$$

In Card-Chess RL, the agent only acts on rounds where it wins the coin toss. The number of game rounds between consecutive agent decisions is a random variable $\Delta t \sim \text{Geometric}(0.5)$. We adapt the shaping formula to:

$$r_{\text{shape}} = \gamma^{\Delta t} \cdot \Phi(s_{t+1}) - \Phi(s_t)$$

This preserves the policy-invariance guarantee of the original theorem while correctly discounting over the variable gap. To our knowledge, this specific adaptation has not been formalised or tested in the RL literature.

### 13.3 Game Design Novelty

Card-Chess RL itself is a novel game variant that does not exist on any current chess platform (Lichess, Chess.com, Fairy-Stockfish). The concept of a shared communal deck of rule-modifying cards drawn without replacement, combined with a coin-toss picker mechanism, creates a game-theoretic structure that has not been previously studied.

### 13.4 Empirical Novelty

The project produces the first empirical comparison of learned vs. heuristic vs. random card-selection strategies in a rule-varying board game, analogous to — but structurally different from — the draft-strategy comparisons in CCG literature (Vieira et al., 2019). The key difference is that in CCGs, drafted cards compose a deck used in a *subsequent* game with fixed rules, whereas in Card-Chess RL, the selected card *immediately and temporarily modifies the rules of the current game*.

---

## 14. References

1. Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., Lillicrap, T., Simonyan, K., & Hassabis, D. (2018). A General Reinforcement Learning Algorithm that Masters Chess, Shogi and Go through Self-Play. *Science*, 362(6419), 1140–1144.

2. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping. *Proceedings of the 16th International Conference on Machine Learning (ICML)*, 278–287.

3. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. *arXiv preprint arXiv:1707.06347*.

4. Tesauro, G. (1995). Temporal Difference Learning and TD-Gammon. *Communications of the ACM*, 38(3), 58–68.

5. Browne, C. B., Powley, E., Whitehouse, D., Lucas, S. M., Cowling, P. I., Rohlfshagen, P., Tavener, S., Perez, D., Samothrakis, S., & Colton, S. (2012). A Survey of Monte Carlo Tree Search Methods. *IEEE Transactions on Computational Intelligence and AI in Games*, 4(1), 1–43.

6. Moravčík, M., Schmid, M., Burch, N., Lisý, V., Morrill, D., Bard, N., Davis, T., Waugh, K., Johanson, M., & Bowling, M. (2017). DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker. *Science*, 356(6337), 508–513.

7. Vieira, R., Tavares, A. R., & Chaimowicz, L. (2019). Drafting in Collectible Card Games via Reinforcement Learning. *Proceedings of the 18th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)*, 1865–1867.

8. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

9. Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harber, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 1928–1937.

10. Nair, A., Srinivasan, P., Blackwell, S., Alcicek, C., Feber, R., De Freitas, N., Panneershelvam, V., Suleyman, M., Beattie, C., Petersen, S., Legg, S., Mnih, V., Kavukcuoglu, K., & Silver, D. (2015). Massively Parallel Methods for Deep Reinforcement Learning. *arXiv preprint arXiv:1507.04296*.

11. Fairbanks-Stockfish Contributors. (2021). Fairy-Stockfish: A Multi-Variant Chess Engine. https://fairy-stockfish.github.io/

12. Leela Chess Zero Contributors. (2018). Leela Chess Zero. https://lczero.org/

---
