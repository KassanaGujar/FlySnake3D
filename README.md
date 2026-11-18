# FlySnake3D

**Can evolutionary optimisation of a real fruit-fly connectome produce a Snake controller that outperforms A\* once the snake is long enough for greedy pathfinding to trap itself?**

This is a research project, not a game demo. The controller is the **FlyWire whole-brain connectome**: 138,639 neurons and 15.1 M connections, simulated with the leaky integrate-and-fire model of Shiu et al. (2024). It is not a hand-designed neural network. Evolution adjusts a small, recorded set of parameters (sensory gains, thresholds, selected synaptic scalings, decoder gains). The wiring diagram itself stays the biological reference.

```
3D Snake ─▶ virtual eye ─▶ visual encoder ─▶ optic-lobe output neurons ─▶ connectome ─▶ descending neurons ─▶ motor decoder ─▶ snake
           (engineered)    (engineered)       (T4/T5, LPLC2, LC4 …)       (biological)   (DNa01/02, DNp01 …)   (engineered)
```

## Status

| stage | status |
|---|---|
| 0. Choose a simulator; prove the connectome can be stimulated, simulated and read | ✅ done: [report](docs/RESEARCH_REPORT_00_simulator_selection.md) |
| 1. Headless Snake core + Godot 4 replay viewer | ✅ done: [viewer](viewer/README.md) |
| 2. A\* baseline ladder (greedy → time-aware → tail check → space check) | ✅ done: [report](docs/RESEARCH_REPORT_01_astar_baselines.md) |
| 1c. Fly brain activity view (138,639 neurons, real recorded spikes) | ✅ done |
| 3–4. Visual encoder / motor decoder connected to the brain | ✅ done |
| 5. Evolution (fly, ANN and shuffled-wiring controls, held-out evaluation) | ✅ done: [report](docs/RESEARCH_REPORT_02_evolution.md) |
| 6. Body-vs-wall vision, food-per-life scoring, repeats, 50×50 test | ✅ done: [report](docs/RESEARCH_REPORT_03_body_vision.md) |
| 7. Dopamine learning, smell, mushroom-body vision (with a matched no-learning control) | ✅ done: [report](docs/RESEARCH_REPORT_04_learning.md) |
| 8. Experiments, analysis, dashboard | |

Headline findings from Milestone 0 (details in the report):
* The GPU simulator reproduces the published Brian2 model (per-neuron firing-rate r = 0.97).
* Looming detectors LPLC2/LC4 drive the escape giant fiber DNp01. Object detectors drive the steering neurons DNa01/DNa02 **contralaterally**.
* Direct photoreceptor stimulation **does not propagate** to the central brain in this LIF model. The retina and lamina use graded, sign-inverting signalling that the model does not capture. Visual input therefore enters at the optic-lobe output neurons.
* Cost: about 3.4 GPU-seconds per simulated brain-second. One million generations at the planned evaluation budget is about 866 years on a GTX 1660 Ti, so the plan measures learning curves over achievable budgets and extrapolates from them.

Headline findings from Stage 2: greedy A\* (the primary baseline) dies by self-trapping in 100/100 games on 20×20, mostly at length 50–100, eating 64 food on average. Adding the virtual-snake tail check raises this to 345 food (87 % of the board).

Headline findings from Stage 5: evolution works (the evolved interface beats the un-evolved one on 64 of 64 unseen games), but the fly plays like a beginner. It eats 18 food per 300 moves with 3.7 deaths, against 22 food and 2.6 deaths for a small neural network on the same inputs, and dies at about 7 cells, long before it could self-trap. On shuffled wiring no visual signal reaches the steering neurons at all. Next steps are in [PROGRESS_AND_NEXT_STEPS.md](docs/PROGRESS_AND_NEXT_STEPS.md).

Headline findings from Stage 6: letting the fly tell its own body from a wall, and scoring food per life, both cut deaths (to 1.3–2.3 per 300 moves over three seeds) but also cut food (13–15), so the fly is not a better player overall. 95–98 % of its crashes are still avoidable and it still dies at 7–9 cells. The same vision makes the small network nearly deathless in short games, and 94 % of its full-game deaths become self-traps, so the long-length A* question is now testable for the network but not yet for the fly.

Headline findings from Stage 7: the fly brain can now learn during play (dopamine-driven Kenyon cell → MBON plasticity on 62,261 FlyWire synapses), smell food, and see through the mushroom body, on a disclosed calibration of the model (adaptation, sparser Kenyon cells) that stops an olfactory lock-up. Learning did **not** help: the learning fly and an identical fly with learning switched off are within each other's confidence intervals on every test (about 20 food and 2 deaths per 300 unseen moves; 8–9 food per full game against 62 for greedy A\*). Both beat the stage 6 fly, so the gain came from the longer readout, smell and extra vision. Crashes are still nearly all avoidable, now with consistent (97 %) decisions.

![evolution results](docs/images/evolution_results.png)

![greedy A* trapped, in the Godot replay viewer](docs/images/viewer_orbit_trap.png)

**To run the viewers:** double-click `run_viewer.bat` (game replays), run `run_viewer.bat --brain` (fly brain), or `run_viewer.bat --brain=results/brain/fly_champion` (the evolved fly playing, with every spike). See [PROGRESS_AND_NEXT_STEPS.md](docs/PROGRESS_AND_NEXT_STEPS.md).

## Quick start (Milestone 0)

```bash
python -m venv ~/.venvs/flysnake3d            # keep the path short on Windows
~/.venvs/flysnake3d/Scripts/python -m pip install -r requirements.txt
~/.venvs/flysnake3d/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
~/.venvs/flysnake3d/Scripts/python scripts/fetch_data.py      # ~140 MB, hash-checked
~/.venvs/flysnake3d/Scripts/python -m pytest
~/.venvs/flysnake3d/Scripts/python -m experiments.pathway_probe
~/.venvs/flysnake3d/Scripts/python -m experiments.retina_propagation
~/.venvs/flysnake3d/Scripts/python -m experiments.validate_backends --trials 4
~/.venvs/flysnake3d/Scripts/python -m experiments.benchmark
```

Results are written to `results/milestone0/`. The committed copies are the runs reported in the docs.

## Layout

```
brain/        connectome_loader.py, neural_simulator.py   (implemented)
experiments/  validate_backends, pathway_probe, retina_propagation, benchmark
scripts/      fetch_data.py
tests/        test_brain.py
docs/         research reports, DEVLOG, data provenance
game/ agents/ evolution/ visualization/   (later stages)
```

## Scientific framing

This is evolutionary optimisation of a *connectome-derived controller*, not biological evolution of flies. The encoder and decoder are engineered. The neural dynamics are a simplified model. A higher score alone will not be read as the fly brain having "learned" anything. Behaviour and parameter changes will be analysed. See §10 of the report for what has not been established yet.

## Credits and licences

Connectome: FlyWire Consortium (Dorkenwald et al. 2024). Annotations: Schlegel et al. 2024. Neural model: Shiu et al. 2024 (MIT). Data provenance and hashes: [docs/DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md). This repository's code is under the MIT licence.
