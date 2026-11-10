# Research report 01: A* baselines and self-trapping

Date: 2026-11-09. Code: `agents/astar_agent.py`, `game/environment.py`, `game/runner.py`, `experiments/astar_baseline.py`. Raw data: `results/stage2/`.

## 1. Where the A* code comes from

The brief asked for "the original A*". No author code was supplied, so the user asked me to look at current A* Snake implementations and write the baseline myself. Everything in `agents/astar_agent.py` was written for this project. Nothing was copied.

The design follows the best-known open implementation, [chuyangliu/snake](https://github.com/chuyangliu/snake). Its graph-search ("greedy") solver:

1. finds the shortest path to the food;
2. sends a *virtual snake* along that path, and accepts the path only if the virtual snake can still reach its own tail afterwards;
3. otherwise follows the *longest* path to its tail;
4. as a last resort, takes the move that heads away from the food.

Most other public Snake A* repositories implement only step 1 on a static body snapshot. That is the version people mean when they say "A* traps itself".

Hamiltonian-cycle solvers are deliberately excluded. They fill the board by never taking a risk. They are not pathfinding, and comparing a fly against them answers a different question.

## 2. The ladder

Each rung adds exactly one idea to the previous one. Comparing adjacent rungs shows what each idea is worth.

| id | class | idea added |
|---|---|---|
| A0 | `AStarGreedy` | Textbook A* to the food. The current body is a static wall. **The primary baseline.** |
| A1 | `AStarTimeAware` | Body occupancy over time: the segment *i* places from the tail is free once the path reaches it at step ≥ *L − i*. |
| A2 | `AStarTailCheck` | A1, plus the virtual-snake tail check, longest-path tail chasing, and a max-space fallback. |
| A3 | `AStarSpaceCheck` | A2, plus a flood-fill veto on every move: never enter a region smaller than the body. |

All four are deterministic. Ties in A* are broken by insertion order. Each agent's runs are tested for exact reproducibility (`tests/test_astar.py`).

## 3. Method

| setting | value |
|---|---|
| Board | 20 × 20 (400 cells), no obstacles |
| Start | length 3, centre, heading east |
| Episodes | 100 seeds (0–99). The same seeds are used for every agent, so food sequences match until the agents' paths diverge. |
| Starvation cap | 400 moves without food |
| Episode cap | 20,000 moves |

**Trap proxy.** Before each move we flood-fill from the head, treating the tail cell as free. A state counts as *trapped* when the reachable area is smaller than the snake's length. This proxy is conservative: the tail moving can still open an escape. So it is always reported together with whether death actually followed (`self_trap_death`) and how many moves later (`moves_trap_to_death`).

## 4. Results

| agent | food, mean [95% CI] | median | max length | board fill | deaths (self / wall) | self-trap deaths | trapped → death, median moves | ms / decision |
|---|---|---|---|---|---|---|---|---|
| A0 greedy | 63.9 [60.7, 67.0] | 64 | 105 | 16.5 % | 87 / 13 | 100 % | 36 | 0.17 |
| A1 time-aware | 68.7 [65.0, 72.2] | 70.5 | 116 | 17.7 % | 87 / 13 | 100 % | 147.5 | 0.17 |
| A2 tail check | **345.3** [340.1, 349.6] | 350 | 379 | 87.1 % | 5 / 0 | 5 % | n = 5 | 0.98 |
| A3 space check | 223.5 [201.1, 246.1] | 159 | 371 | 56.5 % | 52 / 4 | 56 % | 1563.5 | 2.07 |

A2 did not die in 95 episodes: 91 reached the 20,000-move cap and 4 starved. A3 reached the cap in 42 episodes and starved in 2.

Deaths by length at death (`by_length.csv`):

| agent | 0–25 | 25–50 | 50–100 | 100–150 | 150–200 | 200–300 | 300+ |
|---|---|---|---|---|---|---|---|
| A0 | 1 | 11 | 85 | 3 | | | |
| A1 | 1 | 11 | 83 | 5 | | | |
| A2 | | | | | | 2 | 3 |
| A3 | | | 7 | 35 | 13 | | 1 |

## 5. What this shows

1. **Greedy A* self-trapping is reproduced.** A0 dies in every episode. Most deaths come at length 50–100, only 13–25 % of the board, and the trap proxy fires before every one of them. The median lead time is 36 moves. The snake is doomed well before it hits anything, which is the "fills its own space" failure the brief describes.
2. **Time-aware occupancy alone barely helps** (+7 % food). It delays death after the trap (median 148 vs 36 moves) but does not avoid the trap.
3. **The tail check is the idea that matters.** It gives 5.4× more food than A0. A2 is the strong baseline the fly will also be compared against.
4. **A3 is worse than A2.** This is a measured result, not what I expected. The per-move space veto removes the moves A2 uses to chase its tail through tight gaps, so A3 is pushed into the max-space fallback. We keep A3 in the ladder and report it as it is, because dropping a rung after seeing its result would be cherry-picking.

## 6. Limitations

- **The step cap binds for A2.** 91 % of its episodes stopped at 20,000 moves, so its food score is a lower bound. Before the final comparison, either raise the cap or score food per move.
- **The 100 % self-trap figure for A0/A1 is partly a proxy effect.** At the moment of a self-collision the head is nearly always enclosed. The informative number is the lead time (36 moves), not the percentage.
- **The trap proxy can misfire both ways.** A trapped state can be escaped once the tail moves, so a "trap" is not proof of doom. It is reported together with the actual outcome.
- **Only one board size so far.** 50 × 50 generalisation (brief §11) is still to do.
- **The game is 2D.** "3D Snake" here means a 3D *rendering* of a planar game (decided in report 00). The planners would need extending for a volumetric arena.

## 7. Reproduce

```bash
python -m pytest tests/test_game.py tests/test_astar.py
python -m experiments.astar_baseline --episodes 100 --size 20
```

Replays of self-trap deaths for A0, A1 and A3 are committed in `results/replays/stage2/`. A2's replays are 11–48 MB each and are not committed; rerun to regenerate them. Every replay re-simulates exactly (`game.runner.verify_replay`).
