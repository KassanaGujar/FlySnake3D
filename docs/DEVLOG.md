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
