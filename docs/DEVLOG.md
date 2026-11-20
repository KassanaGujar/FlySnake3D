# Development log

## 2026-11-08: Milestone 0: simulator selection and stimulate/simulate/read proof

**Done**
- Compared five candidate foundations by reading their source. Selected the Shiu et al. (2024) LIF model on FlyWire v783. See [RESEARCH_REPORT_00](RESEARCH_REPORT_00_simulator_selection.md).
- `scripts/fetch_data.py`: downloads the connectome and annotations, pinned to commits and checked by SHA-256.
- `brain/connectome_loader.py`: signed synapse matrix `W[post, pre]` and cell-type queries. An unknown type raises an error and never returns an empty result.
- `brain/neural_simulator.py`: Brian2 reference backend plus a batched PyTorch/CUDA backend with the same equations.
- Validation: Torch vs Brian2 firing-rate correlation r = 0.97.
- Probes: the visual-projection → descending pathways respond (LPLC2/LC4 → giant fiber; LC9/LPLC1 → contralateral DNa01/DNa02). **Photoreceptor input does not propagate** past the lamina/medulla in this LIF model.
- Benchmark: 3.4 GPU-s per simulated brain-second (B=64, GTX 1660 Ti). The naive projection for 10⁶ generations is ~866 years.
- 7 unit tests passing.

**Decisions**
| decision | alternatives | why |
|---|---|---|
| Shiu LIF on FlyWire v783 | FlyBrain (snedea), flyvis, FlyGym, eonsystems | the only published, validated whole-brain model with per-neuron I/O |
| Own PyTorch backend (MIT) | use eonsystems' backend (GPL-2.0) | licence freedom and batch-of-genomes design; validated against Brian2 |
| Feature-level visual entry (T4/T5, LC/LPLC) as the primary encoder | photoreceptor entry | photoreceptor entry is measured to be silent downstream; it is kept as a control |
| Python headless game, Godot as viewer only | game logic in Godot | determinism, speed, no rendering in the training loop |

**Open issues**
1. The Torch backend gives ~11% more total spikes than Brian2 (4 trials). Rerun with ≥ 20 trials. If the gap persists, compare per-step event ordering.
2. Retinotopic coordinates for LC/LPLC/T4/T5 are needed (FlyWire column assignment, or synapse positions).
3. Event-driven synaptic update for a 10–50× speedup. Must be revalidated against Brian2.
4. ~~Need the user's original A* code~~. Resolved 2026-11-09: none supplied, so the baseline was written for this project (see report 01).
5. ~~Godot 4 is not installed~~. Resolved 2026-11-10: 4.7.2 installed (checksum verified).

**Next:** Milestone 1, a headless Snake core with tests and a replay format, followed by the Godot viewer.

## 2026-11-09: Stages 1–2: headless Snake core and A* baseline ladder

**Done**
- `game/environment.py`: deterministic Snake with relative actions (left/straight/right). The tail vacates before the collision check. Starvation and step caps. Food is placed uniformly over free cells from a seeded RNG.
- `game/runner.py`: episode runner with per-decision measurements (reachable area, trap proxy, nearest body segment) and JSONL replays that are verified by re-simulation.
- `agents/astar_agent.py`: A0 greedy, A1 time-aware, A2 tail check, A3 space check. Written for this project, modelled on chuyangliu/snake's graph-search solver. No author code was supplied.
- 100 seeded games per agent on 20×20: A0 64 food, A1 69, A2 345, A3 224. A0 self-traps in every game, with a median 36 moves between first trap and death. See [RESEARCH_REPORT_01](RESEARCH_REPORT_01_astar_baselines.md).
- 20 game and A* tests passing (27 total).

**Decisions**
| decision | alternatives | why |
|---|---|---|
| A0 greedy static-body A* is the primary baseline | use the strongest planner only | it is the planner the hypothesis is about; the stronger rungs are reported too so a win over A0 is not overclaimed |
| Keep A3 even though it underperforms A2 | drop it | dropping a rung after seeing its result would be cherry-picking |
| No Hamiltonian solvers | include one | not pathfinding; trivially fills the board by never taking risks |
| Commit replays except A2's | commit all / none | A2's replays are 11–48 MB each |

**Open issues**
6. A2 hits the 20,000-move cap in 91 % of games. Raise the cap or score food per move before the final comparison.
7. Run the ladder on 50×50 for the generalisation test.

**Next:** Godot 4 viewer (needs Godot installed), then retinotopy for the visual encoder.

## 2026-11-10: Stage 1b: Godot replay viewer

**Done**
- Godot 4.7.2 standard build downloaded from the official GitHub release. SHA-512 matched the published sums. Installed to `C:\Users\User\tools\godot-4.7.2`.
- `viewer/`: a 3D replay viewer.
  - Draws the arena, the snake (multimesh), the food and A*'s planned path.
  - Four cameras: orbit, top-down, chase and head position.
  - Playback controls, a timeline slider, a trap banner and a jump-to-trap key.
- Headless `--check` mode validates a replay (frame count, steps, bounds, body contiguity, outcome) and draws every frame. `--screenshot` renders a PNG.
- `tests/test_viewer.py` (28 tests total, all passing). Screenshots in `docs/images/`.

**Decisions**
| decision | alternatives | why |
|---|---|---|
| Build the scene in GDScript code, minimal .tscn | hand-built scenes | diffable, reviewable, no editor needed |
| Discrete frames, no interpolation | smooth tweening | the viewer must show exactly what was recorded |
| Head camera labelled "position only" | call it the fly's view | the brain's input comes from the Python encoder; claiming otherwise would mislead |

**Open issues**
8. Live view over WebSocket; fly-brain overlays.

**Next:** retinotopic coordinates for LC/LPLC/T4/T5 (open issue 2), then the visual encoder.

## 2026-11-10: Stage 1c: fly brain activity view

**Done**
- `experiments/record_brain_activity.py` records every spike of the LIF model, in 2 ms bins:
  - 0–50 ms: silence.
  - 50–250 ms: LPLC2 + LC4 on the left at 150 Hz.
  - 350–550 ms: LC9 on the right at 150 Hz.
  - 700 ms of brain time takes 18.6 s on the GPU.
- Neuron positions come from the FlyWire anchor points (`pos_x/y/z`, 4×4×40 nm voxels, as documented in the flywire_annotations README); 138,625 of 138,639 neurons have one. Output goes to `results/brain/`, which is gitignored and regenerated by the launcher.
- `viewer/brain.gd`: all neurons drawn as a multimesh, coloured by super class. Neurons glow when they spike, then fade. The view shows input status, the most active cell types, per-group spike bars and labels.
- Headless `--brain --check` mode confirms all 20,223 spiking neuron-bins play back.
- `run_viewer.bat`: one-click launcher.
- Progress and the plan are written up in [PROGRESS_AND_NEXT_STEPS.md](PROGRESS_AND_NEXT_STEPS.md).

**Open issues:** labels overlap; base colours are too bright; **verify left/right orientation**; add a brain-view test to pytest.

## 2026-11-11: Brain view polish and orientation check

**Done**
- **Orientation verified from landmarks:** antennal lobe anchors at z ≈ 1,300–1,900 voxels (anterior), Kenyon cells at z ≈ 4,600 and low y (posterior, dorsal), and left-annotated neurons at lower x. The exported frame (+x right, +y dorsal, −z anterior) is right-handed, so the view is not mirrored.
- The default view is from behind (fly's left = screen left). The view previously labelled "front" was really this one. Added a true front view (key 4). The HUD states the viewpoint.
- Silent neurons dimmed to 45 %. Labels are shown only while their group fires, which removes the overlap.
- `tests/test_brain_view.py`: orientation and extent checks, plus headless playback of every spike. 30 tests pass.

**Observation (not yet a result):** in the recording, LC9 on the right is followed by spikes in DNa01 left and DNa02 right. That matches the contralateral DNa01 response from Milestone 0. It is one seed, so it needs repeats before it is relied on for the decoder.

**Open issues:** A2 step cap; hand-check of the interactive controls.

## 2026-11-12: first full fly evolution run, and why it stalls

- `fly_main`: 60 generations, population 64, 300 moves each, about 90 s per generation. Mean fitness -6.9 to about 14 by generation 20, flat afterwards.
- Wrote `experiments/diagnose_deaths.py`. Findings: all 384 crashes avoidable at mean length 5; 70% are turns into an adjacent obstacle; resetting the brain each move trades body crashes for wall crashes; identical views give the same action 89% of the time; 48 ms window does not reduce deaths. Full table in `docs/PROGRESS_AND_NEXT_STEPS.md`.
- Decision: cancelled the shuffled-wiring control until the fly setup is fixed. A control on a broken setup would not be interpretable.
- Added `--death_penalty` and `--init_from` to `experiments/evolve.py` (defaults unchanged, so `fly_main` and `ann_main` are reproducible).
- Uncertain: whether the 54-gene interface can reach ANN-level safety at all. The ANN has 131 free weights and no spike noise.

## 2026-11-15: training setups screened, shuffled control, held-out evaluation

- Screened death penalty 2 (`fly_v2`), late readout (`fly_v3`), 36 ms window (`fly_v4`, cut at gen 13 by sleep) and evenly spaced input (`fly_v5`), each from the `fly_main` champion. Deaths per 300 unseen moves 6.0 -> 4.7 / 4.2 / 5.3 / 3.3.
- Added `regular=True` to `EventLIF.set_poisson_input` (phase accumulator, its own RNG so the Poisson stream is unchanged; tests pass). Deviation from Shiu et al., documented in Report 02.
- Shuffled wiring with v5 settings: every candidate always goes straight. `experiments/shuffled_reach_probe.py`: 0 readout spikes on shuffled vs 143-150 on real wiring (3 seeds).
- `fly_v5_scratch` (v5 settings from scratch, 40 gens) reached only about 15 food and 5 deaths. A harsh penalty from the start slows learning.
- Ran `experiments/evaluate_evolution.py --fly fly_v5_regular_late12 --shuffled fly_v5_shuffled`; results in Report 02. Fixed the viewer path in docs: `--brain=` paths are relative to the repo root.
- The computer slept for about 18 h during the screening; runs since then use the app's keep-awake.

## 15 Nov 2026 (late): stage 6 first results (body-vs-wall vision + food-per-life scoring)

Run `fly_v6_body_perlife` (40 gens, 600-move budget, from the v5 champion) vs `ann_v6_body_perlife`.
Held-out, 64 seeds, 300 moves (fitness = food - 0.5 x deaths for all):
- fly v6: 13.7 food, 1.47 deaths (v5 was 18.0 / 3.7). Fewer deaths but less food; held-out
  fitness peaked at gen 8 (17.2 food / 2.8 deaths) and drifted down after, so per-life scoring
  traded food for survival.
- ANN v6: 20.7 food, 0.02 deaths. Reflex: 17.1 / 0.39.
Full games 20x20: fly 8.6 food (v5 4.5), 206 steps, self-trap deaths 11%; ANN 36.2 food (self-trap 94%);
reflex 29.7; greedy A* 61.7.
50x50 (never trained): fly 9.5 food, ANN 69.0, reflex 68.7.
Diagnosis (seeds 700000+): 85 of 87 crashes still had a safe move; 53 were turns into an adjacent
obstacle; mean length at death 6.6. Still a steering failure, not self-trapping.
Caveat: vision and scoring changed together. Ablation (`fly_v6_body_penalty2`) and repeats
(seeds 1, 2) queued in scripts/stage6_queue.sh; outputs in results/stage6_queue.log.
Note: the ANN with body vision now dies of self-trapping 94% of the time, so the long-length
question is now testable for the ANN control, but not yet for the fly.

## 16 Nov 2026: stage 6 complete

Repeats (seeds 1, 2) and the separation test finished; see Report 03. Unseen-game results:
seed 1 14.3 food / 2.3 deaths, seed 2 15.2 / 1.3; new vision + old scoring 16.7 / 2.3.
Mistake and fix: pausing stopped the training process but not the queue script, which then
diagnosed the half-trained separation run and later skipped it as done. The partial diagnosis
was renamed `_PARTIAL_gen10`, the run resumed from its checkpoint, and the queue now counts a
run as done only when history.csv has all 40 generations.

## 16 Nov 2026 (night): steering decisions rest on about 3 spikes

`experiments/readout_spike_probe.py` on the stage 5 champion (64 unseen games, 100 moves):
mean 3.2 readout spikes per decision (median 3; 61% of decisions have 3 or fewer; 5% have none),
summed over all five descending types and both sides. 89% of decisions are within one weighted
spike of the turn threshold, and one DNp09 spike (weight 0.93 vs threshold 0.82) is enough to
turn. So each move is close to a coin toss on a few spikes. This explains the ~0.83 decision
agreement and the avoidable over-turning crashes better than anything tested so far, and it was
missed: the 24 ms window (12 ms read) is far shorter than the time real descending neurons
integrate over. Candidate fixes: much longer brain time per move, more readout neurons, or
averaging several copies of the brain per decision. Results: results/stage6/readout_spike_probe_*.json

## 16-17 Nov 2026: stage 7 started (dopamine learning + smell); olfactory lock-up found

User direction: the fly should learn within its life through its own dopamine system, smell
should be added alongside vision, and the readout should be fixed. Built (all off by default,
42 tests pass, earlier results unchanged):
- `EventLIF.enable_plasticity`: per-brain scales on all 62,261 Kenyon cell -> MBON synapses;
  dopamine x recent-KC-activity weakens them, slow recovery to the published strength. Which
  dopamine neuron acts on which MBON = its FlyWire synapse count onto that MBON (1,990 pairs).
  Checked: rate 0 gives spike-for-spike the same result as no plasticity.
- `brain/mushroom_body.py`: food odour at left/right antenna -> ORNs of 6 food-odour glomeruli;
  the 8 visual types with most synapses onto KCs (aMe12, MTe32, ...) driven by the visual
  features (the stage 3-6 LC/LPLC types have 1 synapse onto KCs in total); eating -> PAM,
  crashing -> PPL1. FlyPolicy(learning=True), 162 genes; ANN control gets smell too (195).
- Rollout tells policies what happened after each move (`feedback`) and resets learning at the
  start of each evaluation (`new_evaluation`).

Probes (experiments/learning_pathway_probe.py, experiments/olfactory_lockup_probe.py):
- smell -> PN -> KC -> MBON -> readout DNs carries signal; vision (LC types) -> KC does not.
- BLOCKER: in the unmodified LIF model any olfactory input (even 2 Hz into DM1 alone, all 3
  seeds) switches the olfactory system (ORNs via feedback, AL local neurons, PNs, ~3,840 KCs,
  APL, DPM) into permanent firing at the model's maximum rate (333 Hz). Removing all 379k
  KC->KC synapses does not prevent it. One glomerulus activates almost the same KCs as another
  (overlap ~1.0), so the code carries no odour information, and the learning rule just weakens
  every synapse. Vision never triggers it.
- Spike-frequency adaptation (`EventLIF.enable_adaptation`, not in Shiu et al.) of 2 mV per
  spike with 1 s decay stops the lock-up (0 KCs active 120 ms after input) and leaves
  vision -> DN intact (32 vs 35 spikes), but KC activity stays dense (~2,500 of 5,177) and
  non-specific. Real KC coding is sparse (~5-10% per odour; from memory, needs checking).
- Speed: learning fly ~3.9 s per move for 64 brains at 72 ms per move (plain 1.1 s), mostly from
  the runaway activity; expected to drop once activity is fixed.
Decision needed from the user before continuing: how far to modify the published model.

## 17 Nov 2026: stage 7 calibration chosen (user: "Adaptation + sparse cells"); runs queued

Calibration (experiments/kc_sparseness_calibration.py; both changes are NOT in Shiu et al. 2024
and apply only to stage 7 runs, `calibrated=True`):
- every neuron: spike-frequency adaptation, +2 mV threshold per spike decaying over 1 s;
- Kenyon cells: +8 mV threshold. APL unchanged.
Result: 5-6% of Kenyon cells active per odour (target 5-10%), no activity after the odour stops,
vision -> steering neurons unchanged (32 spikes either way).
Not fixed: one glomerulus already activates ~500 of 685 projection neurons (~150 types) in the
antennal lobe, so different odours still recruit overlapping Kenyon cells (Jaccard ~0.65-0.75).
Odour identity is lost before the mushroom body. Accepted for Snake (one food odour); recorded
as a limitation.

Regression (experiments/calibration_regression.py): the stage 5 champion on the calibrated model
drops from 17.4 food / 3.8 deaths to 13.7 / 6.8 on the same unseen games. Expected (its interface
was evolved without adaptation), and it means the learning test needs a matched control evolved on
the calibrated model, not a comparison with old champions.

Readout fix: 72 ms per move, spikes read from 12 ms (60 ms read). Readout spikes per decision
2.1 -> 6.6 (decisions on 3 or fewer spikes 79% -> 37%); 120 ms gives 10.2 at 1.7x the cost.
Spikes grow more slowly than the window because adaptation tires the descending neurons.
Decoder threshold range doubled (theta_scale 8). Speed: 1.67 s per move for 64 learning brains
(the earlier 3.9 s at 24 ms came from the runaway activity, now gone).

Queued (scripts/stage7_queue.sh, resumable; resume_stage7.bat): ann_v7_odour (ANN + smell),
fly_v7_learn and fly_v7_control (identical, learning forced off), both from the
fly_v6_body_penalty2 champion, death costs 2 food, 600 moves, 40 generations; then diagnosis
and full evaluation of both. About 11 h per fly run.

## 18 Nov 2026: stage 7 results; learning changes synapses but not play

Runs (scripts/stage7_queue.sh): ann_v7_odour, fly_v7_learn, fly_v7_control (identical,
--no_plasticity), each 40 generations (ANN 80) from the fly_v6_body_penalty2 champion.
Generation time 13-17 min on a quiet machine, about 50 min on 17 Nov 8 am-9 pm while Ollama
(llama-server, about 2 cores and GPU memory) and game launchers ran and RAM was full (0.2 of
16 GB free). The brains were not the cause: generation 2 and 17 champions produce the same
activity (63 spikes, 16,000 synaptic events per step per brain). Stuck `remember` plugin
session-start scripts (orphaned, one core each) were stopped by a 10-minute cleanup loop
until the queue finished.

Unseen games (500000-500063, 300 moves): learn 20.3 food / 2.33 deaths, control 19.7 / 1.73,
ANN with smell 21.4 / 0.41. Full games: learn 7.8, control 8.8 (paired -0.9, CI -3.2 to +1.3);
50x50: 17.7 vs 15.3 (+2.4, CI -1.2 to +6.1). Death diagnosis: 99-100 % avoidable, length 9,
97-98 % decision agreement. experiments/learning_ablation_probe.py: same learning champion,
learning on vs off: 19.8 vs 20.2 food (paired -0.34, CI -0.84 to +0.16); about 1.5 % of
plastic synapses (about 930 per brain) weakened by more than 10 % during play.
Evolution lowered the learning rate about 5-fold and recovery about 2.6-fold.
Conclusion: no measurable benefit from learning; gains over stage 6 come from the readout,
smell and mushroom-body vision. Likely reason: MBONs have a weak hold on the steering readout
(902 synapses vs 2,105 from vision). Report: docs/RESEARCH_REPORT_04_learning.md.

## 2026-11-18 (later): stage 8, the final learning trial (user: "last trial before just saying that the fly cannot")

User chose both changes:
1. Per-move teaching signals (engineered, disclosed; not controller rules). After every move:
   cell straight ahead of the head blocked -> PPL1 (punishment); head closer to food without
   eating -> PAM (reward). Two evolved drive genes. Death and food dopamine unchanged.
2. MBON readout: the decoder also reads all 35 FlyWire MBON types (96 neurons, every type
   bilateral), one evolved weight per type on right - left spikes, same form as the DN
   readout. Weights start at 0, so the starting brain plays exactly like the stage 7 champion.
Genome 162 -> 199 (agents/policies.py to_v8_genome). Flags --mbon_readout --move_feedback.

Bug found and fixed while building the probe: FlyPolicy.feedback stored the caller's arrays,
and act() clears them in place, so a second policy fed the same arrays never got dopamine.
Training runs one policy and was not affected; feedback now copies.

Pre-test (experiments/move_feedback_probe.py, 64 brains, 300 moves, seeds 600000+): learning
copy and frozen copy in lockstep on the same games (sanity mode with both frozen: identical).
Learned change in MBON (right - left) spikes summed over types: 4.6 per decision in the first
50 moves, 6.2 in the last 100, against a descending-neuron distance to threshold of 1.7-1.9.
With |w| = 1 learning could flip 71 % -> 90 % of moves. Caveat: the change is as large as the
MBON signal itself (6.4), so part of it may be spike-timing divergence rather than useful
memory; it is an upper bound on influence, not evidence of useful learning.
6.1 danger moves and 260 closer moves per brain per 300 moves (the reward fires on most moves).

Started fly_v8_learn and fly_v8_control (--no_plasticity), both from the fly_v7_learn champion,
40 generations, 600 moves, death penalty 2, all stage 7 settings, running in parallel.
54 tests pass.

Speed: about 27 min per generation each (GPU 91 % busy; together about the same total time as
one after the other). The user agreed to stop at generation 20 if both runs are flat.
Stopping rule, fixed before generation 20 is seen (gens 0-19, population mean fitness):
  flat     = least-squares slope over gens 10-19 below +0.5 fitness per generation, in BOTH runs
  learning ahead = learn minus control, averaged over gens 15-19, more than 2 standard errors
           above 0 (standard error from the 5 per-generation differences)
Stop both at generation 20 only if both are flat AND learning is not ahead; otherwise run to 40.
Either way the champions are evaluated the same way afterwards.

2026-11-18 22:30: both runs were paused by the resource-allocation session at about 22:02 and the PC then rebooted. Resumed from checkpoint.npz with the same command (learn at generation 12, control at 9); new output in results/stage8/*_resume.log. No generation was lost.

2026-11-19 05:34, generation-20 check (pre-registered rule): population mean fitness slope over gens 10-19 learn +0.94, control +0.75 per generation, so NOT flat; learn minus control over gens 15-19 = -2.22 (2 SE = 0.77), i.e. the control is ahead. Rule says continue to 40; both runs continue.

## 2026-11-20: stage 8 results and conclusion

Both runs finished 40 generations (19-20 Nov). Evaluated with evaluate_evolution.py (--stage stage8 / stage8_control, --big, --skip_dt_check) and diagnose_deaths.py --quick; paired comparison in experiments/compare_stage8.py -> results/stage8/comparison.json.
Full games learn - control: 20x20 -2.2 (CI -4.3 to -0.2), 50x50 -7.4 (-10.9 to -4.1). Unseen 300 moves: learn 19.0 food / 2.03 deaths, control 20.1 / 1.89; both equal to their starting genome. Learning champion: danger drive 0 Hz, closer 27 Hz, mean |MBON weight| 0.24 (control 0.15). Learning fly dies on its own body more (69 vs 40 of 150 / 121 deaths).
Conclusion recorded in Report 05: the fly can be evolved to play poorly but not well; learning did not help (stage 7) and hurt (stage 8). 54 tests pass.
