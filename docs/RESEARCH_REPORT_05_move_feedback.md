# Research Report 05: per-move teaching signals and the mushroom-body readout

*FlySnake3D, stage 8, the final learning trial. Written 20 November 2026.*

## Question

Stage 7 found that dopamine learning in the mushroom body changed synapses but not play.
Two reasons were identified: the crash punishment came once, at the end of a life, and
the mushroom body had only a weak path to the steering neurons that decide each move.
Stage 8 fixes both and asks a last time: **does a fly brain that learns during play
beat the same brain that cannot?** The user set this as the last trial before concluding
whether the fly can play.

## What was added

1. **Per-move teaching signals (engineered shaping, disclosed).** After every move:
   - *danger*: the cell straight ahead of the head is blocked, which drives PPL1
     (punishment) dopamine neurons;
   - *closer*: the distance to food dropped without eating, which drives PAM (reward)
     dopamine neurons.

   Each has an evolved strength (0–200 Hz). These are training signals computed by the
   game, not rules in the controller. The brain still chooses every move from its
   spikes alone; no direction, path or A\* information is passed to it.
2. **MBON readout.** All 35 FlyWire MBON types (96 neurons, every type bilateral) are
   added to the decoder with one evolved weight per type on right − left spikes, the
   same form as the descending-neuron readout. Weights start at 0, so the starting
   genome plays exactly like the stage 7 champion.

Genome 162 → 199 genes. Everything else is as in stage 7, including the disclosed
calibration (spike-frequency adaptation, Kenyon cell threshold +8 mV).

## Pre-test: can learning reach the moves now?

`experiments/move_feedback_probe.py`: 64 brains, 300 moves. A learning copy and a frozen
copy see identical inputs and dopamine, so their difference comes from learning alone.
A sanity mode with both frozen gives exactly zero difference.

| | first 50 moves | last 100 moves |
|---|---|---|
| learned change in MBON right − left spikes per move | 4.6 | 6.2 |
| distance of the steering signal from a decision threshold | 1.9 | 1.7 |
| moves learning *could* flip at full MBON weight | 71 % | 90 % |

So the route from learning to moves was open. This is an upper bound: part of the change
may be spike timing drifting apart rather than useful memory.

## Design

| run | what differs |
|---|---|
| `fly_v8_learn` | learning on, per-move signals, MBON readout |
| `fly_v8_control` | identical, synapses never change (`--no_plasticity`) |

Both start from the stage 7 learning champion. 64 brains per generation, 40 generations,
600 moves per evaluation, fitness = food − 2 × deaths, same seeds and code. A stopping
rule fixed in advance (stop at generation 20 if both runs had stopped improving and
learning was not ahead) did not trigger: both runs were still improving, so both ran
to 40. The runs were interrupted twice (a pause, then a reboot) and resumed from their
checkpoints with no generation lost.

## Short answer

- **Learning made the fly worse.** In full games the learning fly eats 2.2 food less
  than the control on 20 × 20 (95 % CI −4.3 to −0.2) and 7.4 less on 50 × 50
  (−10.9 to −4.1). Both intervals exclude zero.
- **Evolution added nothing on unseen games** in either run: both champions score the
  same as the stage 7 champion they started from.
- **Evolution switched the punishment off.** The learning champion's danger drive is
  0 Hz; it kept a weak "closer to food" reward (27 Hz).
- **Still a steering failure:** 97–100 % of crashes had a safe move, and the snake dies
  at 8–9 cells, far below the lengths where A\* traps itself.

## Training

Population averages over the last 5 generations, per 600 moves:

| | fitness | food | deaths | danger moves |
|---|---|---|---|---|
| learning fly | 28.2 | 37.9 | 4.8 | 12.0 |
| control | 30.1 | 39.4 | 4.6 | 11.1 |

At the generation-20 check the control was already 2.2 fitness ahead (2 standard errors
= 0.8).

## Results on unseen games

Seeds 500000–500063, 300 moves, restart on death (`results/stage8*/evaluation.json`).

| agent | food | 95 % CI | deaths |
|---|---|---|---|
| stage 8 learning fly, final | 19.0 | 18.5–19.6 | 2.03 |
| stage 8 learning fly, starting genome | 19.8 | 19.3–20.3 | 1.97 |
| stage 8 control, final | 20.1 | 19.5–20.8 | 1.89 |
| stage 8 control, starting genome | 19.7 | 19.1–20.3 | 2.20 |
| stage 7 learning fly | 20.3 | 19.8–20.9 | 2.33 |
| ANN with smell | 21.4 | 20.8–22.0 | 0.41 |
| hand-written reflex | 17.1 | 16.3–18.0 | 0.39 |

Neither run improved on its starting genome. The rise in training scores came from the
population converging on what the starting champion already did.

## Full games and the 50 × 50 board

One game per seed until death (64 games, paired by seed; `results/stage8/comparison.json`).

| agent | 20 × 20 food | 50 × 50 food |
|---|---|---|
| learning fly | 6.7 | 9.4 |
| no-learning control | 8.9 | 16.8 |
| **paired difference, learn − control** | **−2.2 (−4.3 to −0.2)** | **−7.4 (−10.9 to −4.1)** |
| games where learning did better / worse | 19 / 36 | 13 / 43 |
| ANN with smell | 23.4 | 38.2 |
| hand-written reflex | 29.7 | 68.7 |
| greedy A\* | 61.7 | |
| A\* with tail check | 346.2 | |

For comparison, the stage 7 pair was level: −0.9 (−3.2 to +1.3) and +2.4 (−1.2 to +6.1).

## Why the fly still dies

Death diagnosis (`results/stage5/death_diagnosis_fly_v8_*.json`, 64 unseen games,
300 moves):

| | learning fly | control |
|---|---|---|
| food per brain | 18.9 | 20.1 |
| deaths (all brains) | 150 | 121 |
| wall / own body | 81 / 69 | 81 / 40 |
| crashes with a safe move | 97 % | 100 % |
| length at death | 8.3 | 9.3 |

The learning fly hits its own body far more often (69 against 40).

## What evolution did with the new genes

| | learning champion | control champion |
|---|---|---|
| mean MBON weight size | 0.24 | 0.15 |
| danger (punishment) drive | 0 Hz | 7 Hz (no effect: synapses fixed) |
| closer (reward) drive | 27 Hz | 74 Hz (no effect) |

The learning fly gave the MBONs more weight than the control, so the mushroom body was
used, but evolution turned the per-move punishment off entirely. Learning that changes
the readout mid-game appears to be something evolution had to work around, not exploit.

## Interpretation, and what this does and does not show

- **The two stage 7 explanations were tested and ruled out.** Punishment now arrives at
  the moment of danger, and learned changes reach the decision. The result is that
  learning becomes harmful, not useful.
- A likely reason: the Kenyon cell → MBON rule depresses synapses when dopamine
  coincides with activity. In a fly that pairs an odour with a later outcome over seconds.
  In Snake the same cues (food smell, a wall ahead) recur every few moves in different
  situations, so per-move depression drifts the readout instead of storing "turn away
  from walls". Evolution's answer was to silence the punishment.
- The per-move signals are engineered by the game. They are disclosed as shaping. They
  do not give the brain a path, a direction or an A\* solution.
- One seed per condition. The full-game differences are large relative to their CIs,
  but a repeat would confirm them.

## Conclusion of the learning trials

Across stages 5–8 the FlyWire connectome, driven through our visual encoder and read
through its steering (and now mushroom-body) neurons, **can be evolved to play Snake
poorly but not well**: about 19–20 food and 2 deaths per 300 unseen moves, 7–9 food per
full game against 62 for greedy A\*, death at 8–9 cells from avoidable crashes. A small
neural network and a hand-written reflex both play better. **Within-life dopamine learning
did not help in stage 7 and made play worse in stage 8.** The question the project set,
whether the fly can beat A\* at long snake lengths, cannot be reached: the fly never
grows long enough for A\*'s self-trapping to matter.

This is a result about this interface (engineered encoder, about 200 evolved interface
parameters, fixed wiring) on this model (Shiu et al. 2024 LIF, calibrated). It does not
show that no interface could do better.

## Files

- Code: `brain/mushroom_body.py` (MBON types, `move_signals`), `brain/motor_decoder.py`
  (extra readout groups), `agents/policies.py` (`mbon_readout`, `move_feedback`),
  `evolution/rollout.py`, `experiments/evolve.py` (`--mbon_readout`, `--move_feedback`).
- Probes and analysis: `experiments/move_feedback_probe.py`,
  `experiments/compare_stage8.py`; tests in `tests/test_stage8_move_feedback.py`.
- Results: `results/evolution/{fly_v8_learn,fly_v8_control}`, `results/stage8`,
  `results/stage8_control`, `results/stage5/death_diagnosis_fly_v8_*.json`.
- Figures: `docs/images/evolution_results_stage8.png`, `evolution_results_stage8_control.png`.
