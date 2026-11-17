# Research Report 03: telling body from wall, and scoring per life

*FlySnake3D, stage 6. Written 16 November 2026.*

## Question

Stage 5 ended with the fly dying about 3.7 times per 300 moves, almost always by
steering into something it could have avoided, and never growing long enough to
trap itself. Two changes were tried:

1. **Body-versus-wall vision.** Three extra features per eye (body to the side,
   diagonally and ahead; 1/distance when the first thing seen is the snake's own
   body, 0 when it is a wall). Still only what is visible from the head: no
   path, no plan, no look-ahead. The genome grows from 54 to 78 numbers; old
   genomes load with gain 0 for the new features and behave exactly as before.
2. **Food-per-life scoring.** Fitness = food / (deaths + 1) on a 600-move budget,
   instead of food − 2 × deaths on 300 moves, to reward surviving long enough to
   grow.

## Short answer

- **Neither change made the fly a better player.** Both cut deaths, but both also
  cut food. On the comparable score (food − 0.5 × deaths) the stage 5 champion is
  still the best fly.
- **The separation test splits the effect:** the new vision alone removes about
  one death per 300 moves at the cost of about 1.3 food; per-life scoring removes
  about half a death more and costs about 2.5 more food.
- **The result repeats across seeds**, and the cause of death does not change:
  95–98% of crashes had a safe move, and the snake was 7–9 cells long.
- **The same new vision transforms the ANN control:** almost no deaths in short
  games, 36 food per full game, and 94% of its full-game deaths are now
  self-traps, the situation where greedy A* fails.
- **The fly versus A* at long lengths is still not testable.** It is now testable
  for the ANN control.

## Results on unseen games

Seeds 700000–700063, 300 moves, restart on death. All runs start from the stage 5
champion and evolve for 40 generations.

| run | vision | scoring | food | deaths | length at death | crashes with a safe move |
|---|---|---|---|---|---|---|
| `fly_v5_regular_late12` (stage 5) | old | food − 2 × deaths | **18.0** | 3.3 | 6.8 | 207 of 213 |
| `fly_v6_body_penalty2` (separation test) | new | food − 2 × deaths | 16.7 | 2.3 | 7.5 | 145 of 146 |
| `fly_v6_body_perlife`, seed 0 | new | food per life | 13.2 | 1.5 | 6.6 | 85 of 87 |
| `fly_v6_body_perlife`, seed 1 | new | food per life | 14.3 | 2.3 | 6.8 | 143 of 148 |
| `fly_v6_body_perlife`, seed 2 | new | food per life | 15.2 | **1.3** | 8.7 | 83 of 86 |

The three per-life seeds give food 13–15 and deaths 1.3–2.3, so the trade-off is
real and not one lucky run. The separation test is a single seed.

Caveat on the separation test: it was paused at generation 11 and resumed from
its checkpoint. The checkpoint holds the full search state, so resuming does not
change the run, but the half-finished generation 12 was evaluated twice.

## Full evaluation of `fly_v6_body_perlife` (seed 0)

Held-out short games (seeds 500000–500063, 300 moves; score = food − 0.5 × deaths
for everyone):

| controller | score | food | deaths | longest snake |
|---|---|---|---|---|
| Fly, stage 6 | 12.9 | 13.7 | 1.5 | 21 |
| Fly, stage 5 | 16.1 | 18.0 | 3.7 | 17 |
| ANN, stage 6 vision | 20.7 | 20.7 | 0.02 | 28 |
| Hand-written reflex | 17.0 | 17.1 | 0.4 | 27 |

Held-out score peaked at an early checkpoint (generation 8: 17.2 food, 2.8 deaths)
and then drifted down while deaths kept falling: the per-life score kept paying
for caution after it had stopped paying for food.

Full games, A* benchmark rules (seeds 0–63):

| controller | food per game | moves | self-trap share of deaths |
|---|---|---|---|
| Fly, stage 6 | 8.6 | 206 | 11% |
| Fly, stage 5 | 4.5 | 74 | 2% |
| ANN, stage 6 vision | 36.2 | 576 | 94% |
| Hand-written reflex | 29.7 | 686 | 59% |
| A0 greedy A* | 61.7 | 974 | 100% |
| A2 A* + tail check | 346.2 | 19,473 | 5% |

In a whole game the stage 6 fly does better than stage 5 (it lives almost three
times as long), but it still dies at around 7–9 cells, long before self-trapping
matters.

### 50×50 board (never seen in training)

| controller | food per game | moves |
|---|---|---|
| Fly, stage 6 | 9.5 | 388 |
| ANN | 69.0 | 2,536 |
| Hand-written reflex | 68.7 | 3,186 |

The fly transfers (it eats about as much as on 20×20), but the extra space does
not help it: it still crashes at short lengths. The ANN and the reflex roughly
double their food on the larger board.

## Interpretation

- The fly's limit is **steering reliability**, not what it can see. Giving it
  more information made it more cautious, not more accurate.
- The same features are enough for a 12-input, 8-unit network to play well. So
  the bottleneck is in the path from features through the fixed connectome to the
  five descending neuron types, and in the 78 numbers evolution can tune there.
- This is scientifically uncertain: it could be the interface (per-type gains,
  only five readout types), the noise of spiking decisions (agreement about 0.83
  in stage 5), or the evolution budget. These stage 6 runs do not separate those.

## Limitations

- One seed for the separation test and for the full evaluation; three seeds for
  the per-life setting.
- Everything continues from the stage 5 champion, so these are refinements of one
  lineage, not independent searches.
- The ANN has more free parameters and no spiking noise; it is an upper
  reference, not a like-for-like competitor.
- Readout neurons and their roles still need their citations checked (see
  Report 02).

## What would most likely help next

1. **More interface freedom without touching the connectome:** per-neuron rather
   than per-type gains, and more descending neuron types in the readout.
2. **Average over several brain samples per decision**, or a longer window, to
   reduce decision noise. Measure agreement again.
3. **Run the long-length A* comparison with the ANN control now**, since it does
   reach self-trapping lengths; it shows what the test looks like when the fly
   gets there.
4. **A degree-preserving wiring control** (carried over from Report 02).

## Files

- Runs: `results/evolution/{fly_v6_body_perlife, fly_v6_body_perlife_s1, fly_v6_body_perlife_s2, fly_v6_body_penalty2, ann_v6_body_perlife}/`
- Diagnoses: `results/stage5/death_diagnosis_fly_v6_*.json` (the `_PARTIAL_gen10` file is from the paused run and is not used)
- Evaluation: `results/stage6/evaluation.json`, `episodes.csv`, `episodes_50x50.csv`; figure `docs/images/evolution_results_stage6.png`
- Champion replay and brain: `results/replays/stage6/`, `run_viewer.bat --brain=results/brain/fly_champion_stage6`
- Queue: `scripts/stage6_queue.sh` (resumable; `resume_stage6.bat` to continue it)
