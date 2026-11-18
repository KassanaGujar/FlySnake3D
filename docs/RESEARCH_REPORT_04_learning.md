# Research Report 04: dopamine learning, smell and mushroom-body vision

*FlySnake3D, stage 7. Written 18 November 2026.*

## Question

Until stage 6 every synapse of the fly brain was fixed at its FlyWire strength, and only
the interface (how the game is shown to the brain and how its spikes are read) evolved.
A real fly learns: dopamine neurons change Kenyon cell → MBON synapses in the mushroom
body after reward or punishment. Stage 7 asks whether a fly brain that can learn this way
during its life plays Snake better than the same brain that cannot.

## What was added

All of it uses FlyWire neurons and synapses; no pathway or neuron was invented.

1. **Three-factor plasticity on Kenyon cell → MBON synapses** (62,261 synapses). Each
   synapse has a scale (1 = FlyWire strength) that is depressed when Kenyon cell activity
   coincides with dopamine arriving at that MBON, and recovers slowly towards 1.
   Which dopamine neuron reaches which MBON comes from FlyWire DAN → MBON synapse counts
   (1,990 pairs). Eating food drives PAM neurons (reward), crashing drives PPL1 neurons
   (punishment). Six learning genes (rate, recovery, two trace time constants, reward and
   punishment strength) evolve with the interface.
2. **Smell.** Food odour at the left and right antenna (falls off over 6 cells) drives
   the olfactory receptor neurons of six glomeruli (DM1, DM2, DM4, DP1m, VA2, VM2).
3. **Mushroom-body vision.** The eight visual types with the most synapses onto Kenyon
   cells (aMe12, MTe32, MTe30, LTe25, MTe40, aMe20, aMe26, LTe16) get the same per-eye
   features as the steering pathway.
4. **Readout fix.** 72 ms of brain time per move instead of 24 ms, read from 12 ms:
   readout spikes per decision rose from 2.1 to 6.6.

### Model calibration (disclosed; stage 7 only)

In the unmodified Shiu et al. 2024 model any olfactory input, even 2 Hz into one
glomerulus, locked the olfactory system into permanent firing at 333 Hz with about 3,840
Kenyon cells active (`experiments/olfactory_lockup_probe.py`). The user chose two
calibrations, both NOT in the published model:

- spike-frequency adaptation for every neuron (+2 mV threshold per spike, decaying over 1 s);
- Kenyon cell threshold raised by 8 mV.

Result: 5–6 % of Kenyon cells active per odour, no lingering activity, vision →
steering unchanged. Not fixed: one glomerulus already activates about 500 of 685
projection neurons, so different odours look alike to the mushroom body (Jaccard
0.65–0.75). That does not matter much for Snake, which has a single food odour.

The calibration costs the old stage 5 champion: 17.4 → 13.7 food, 3.8 → 6.8 deaths
(`experiments/calibration_regression.py`). So stage 7 flies are compared with a
**matched control**, not with older champions.

## Design

| run | what differs |
|---|---|
| `fly_v7_learn` | calibrated brain, smell, mushroom-body vision, learning on |
| `fly_v7_control` | identical, learning rate forced to 0 (synapses never change) |
| `ann_v7_odour` | small neural network with the same vision plus smell |

Both flies start from the stage 6 champion (`fly_v6_body_penalty2`), 64 brains per
generation, 40 generations, 600 moves per evaluation, fitness = food − 2 × deaths.
Same seeds, same code, same settings; the only difference is the `--no_plasticity` flag.

## Short answer

- **Learning did not make the fly a better player.** On every test the learning fly and
  the no-learning control are within each other's 95 % confidence intervals.
- **Both stage 7 flies are better than the stage 6 fly** on unseen games (about 20 food
  and 2 deaths per 300 moves against 16.7 and 2.3), but since the control does as well,
  that gain comes from the readout fix, smell and the extra vision, not from learning.
- **Evolution turned learning down but not off:** the population's learning rate fell
  about 5-fold and recovery about 2.6-fold (smaller, longer-lasting changes).
- **Still a steering failure:** 99–100 % of crashes had a safe move; the snake dies at
  about 9 cells, far below where A* self-traps.

## Same brain, learning on vs off

The evolved learning champion replayed on the unseen games with its learning switched
off (`experiments/learning_ablation_probe.py`, paired by seed):

| | learning on | learning off |
|---|---|---|
| food per 300 moves | 19.8 | 20.2 |
| deaths per 300 moves | 2.36 | 2.44 |
| plastic synapses weakened by more than 10 % by the end | 1.5 % (about 930 per brain) | 0 |
| weakened by more than half | 0.06 % | 0 |
| mean weakening | 0.6 % | 0 |

Paired difference (on − off): food −0.34 (95 % CI −0.84 to +0.16), deaths −0.08
(−0.45 to +0.30). **The synapses do change during play, but the play does not.**

## Results on unseen games

Seeds 500000–500063, 300 moves, restart on death (`results/stage7*/evaluation.json`).

| agent | food | 95 % CI | deaths |
|---|---|---|---|
| stage 7 learning fly | 20.3 | 19.8–20.9 | 2.33 |
| stage 7 no-learning control | 19.7 | 19.2–20.3 | 1.73 |
| stage 6 fly (other seeds, uncalibrated) | 16.7 | | 2.3 |
| ANN with smell | 21.4 | 20.8–22.0 | 0.41 |
| hand-written reflex | 17.1 | 16.3–18.0 | 0.39 |

The learning fly eats 0.6 more food but dies 0.6 more often. Evolution added little
beyond the starting genome in either run (learning fly: 18.8 → 20.3 food; control:
17.0 → 19.7).

## Full games and the 50 × 50 board

One game per seed until death (64 games; paired by seed).

| agent | 20 × 20 food | 50 × 50 food |
|---|---|---|
| learning fly | 7.8 (6.3–9.5) | 17.7 (14.4–21.1) |
| no-learning control | 8.8 (7.1–10.6) | 15.3 (12.6–18.2) |
| paired difference, learn − control | −0.9 (−3.2 to +1.3) | +2.4 (−1.2 to +6.1) |
| ANN with smell | 23.4 | 38.2 |
| hand-written reflex | 29.7 | 68.7 |
| greedy A\* | 61.7 | |
| A\* with tail check | 346.2 | |

Neither difference is distinguishable from zero. On 20 × 20 the learning fly does better
on 23 seeds and worse on 26.

## Why the fly still dies

Death diagnosis on the training games (`results/stage5/death_diagnosis_fly_v7_*.json`):

| | learning fly | control |
|---|---|---|
| food / deaths per brain | 19.7 / 1.9 | 19.9 / 1.9 |
| wall / own body | 89 / 33 | 68 / 54 |
| crashes with a safe move | 100 % | 99 % |
| went straight into an adjacent obstacle | 71 % | 50 % |
| length at death | 8.9 | 9.0 |
| same choice from fresh brains, same view | 97 % | 98 % |

The readout fix worked: decisions are now consistent (97–98 % agreement, against
decisions resting on about 3 spikes before). The fly makes the same wrong choice
reliably, so the remaining failure is in what the interface sends and reads, not in
noise.

## Interpretation, and what this does and does not show

- The learning rule as implemented is the textbook one (dopamine depresses
  co-active Kenyon cell → MBON synapses). It works as intended: about 930
  synapses per brain are weakened by more than 10 % within 300 moves, some to zero.
  The mushroom body has a weak hold on the 14 steering readout neurons that decide each
  move: MBONs make 902 synapses onto them against 2,105 from the visual input, and
  driving every Kenyon cell at 50 Hz gives the readout about 50 spikes against about 150
  from vision (`results/stage7/learning_pathway_probe.json`, unmodified model, 240 ms). In the real fly the
  mushroom body biases behaviour over seconds to minutes (approach or avoid an odour),
  not millisecond-scale steering, which fits what we see.
- Snake punishes a crash once, at the end of a life, and the cue that predicted it
  (a wall straight ahead) is seen through the steering pathway, not the mushroom body.
  The credit assignment the game needs (this turn, at this wall) is not what this
  circuit does.
- This is one seed per condition. A repeat would tighten the comparison, but the
  differences are small compared with the gap to the ANN and to A\*.

## Files

- Code: `brain/mushroom_body.py`, `brain/event_simulator.py` (adaptation, plasticity),
  `agents/policies.py` (`learning=True`), `experiments/evolve.py` (`--learning`,
  `--no_plasticity`, `--odour`).
- Probes: `olfactory_lockup_probe.py`, `kc_sparseness_calibration.py`,
  `calibration_regression.py`, `readout_spike_probe.py`, `learning_pathway_probe.py`,
  `learning_ablation_probe.py`.
- Results: `results/evolution/{fly_v7_learn,fly_v7_control,ann_v7_odour}`,
  `results/stage7`, `results/stage7_control`.
- Figures: `docs/images/evolution_results_stage7.png`, `evolution_results_stage7_control.png`.
