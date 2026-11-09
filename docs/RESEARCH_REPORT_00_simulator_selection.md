# Report 00: Choosing a Fly-Brain Simulator, and Proving It Can Be Stimulated and Read

*Milestone 0 · 2026-11-08 · status: complete*

This report answers the project's first assignment. It compares the available FlyWire-based simulators and picks one. It then verifies, by running it, that the chosen connectome can be **stimulated**, **simulated**, and **read**. Finally, it measures what that costs. Every number below comes from a script in this repository. The script and its output file are named beside each result.

---

## 1. Candidates examined

I cloned and read the source of each candidate instead of relying on its README.

| | **Shiu et al. LIF** ([philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model)) | **FlyBrain** ([snedea/flybrain](https://github.com/snedea/flybrain)) | **eonsystems fly-brain** ([eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)) | **flyvis** (Lappalainen et al. 2024) | **NeuroMechFly v2 / FlyGym** (Wang-Chen et al. 2024) |
|---|---|---|---|---|---|
| Connectome | FlyWire FAFB **v783** (and v630). 138,639 neurons, 15.1 M connections, 54.5 M synapses | FlyWire v783 (Codex export) | FlyWire v783 (same files as Shiu) | Optic lobe only, from FIB-SEM column reconstructions (**not FlyWire**) | Body and biomechanics. The brain is user-supplied |
| Whole brain? | Yes (central brain + optic lobes) | Yes, but its own SPEC.md describes a "simplified connectome (~50-80 functional neuron groups)" | Yes | No: retina → T4/T5 motion pathway | n/a |
| Neuron model | Current-based LIF with exponential synapse. Parameters taken from literature, plus one fitted free parameter (`w_syn`). **Published and validated** (Nature 634:210, 2024) | Discrete-time "leaky" units: `V *= 0.95`, threshold 1.0, weights normalised by the max weight. No time units, no published validation | Ports of the Shiu model (Brian2CUDA, PyTorch, NEST GPU, GeNN) | Graded-potential, connectome-constrained deep network trained on motion tasks | Physics (MuJoCo) |
| Simulator | Brian2 (Python) | JavaScript Web Worker | Several | PyTorch | MuJoCo + Gymnasium |
| Sensory input | Poisson "optogenetic" activation of **any neuron by ID** | Hand-picked groups (Feed/Touch/Air/Light buttons) | Same as Shiu | Rendered images on a hexagonal eye | Rendered eye images, odour |
| Motor output | Spike monitor on **any neuron** | Hand-assigned "motor groups" | Same as Shiu | T4/T5 and other cell activities | Joint actuators. Descending drive is user-defined |
| Licence | MIT | MIT | GPL-2.0+ | MIT | Apache-2.0 |

**Verdict.** Only the Shiu et al. model is all three of: (a) the whole FlyWire brain, (b) a published, experimentally validated neural model, and (c) an interface that can address individual neurons. FlyBrain is an attractive visual demo. Its dynamics, however, are ad hoc and unvalidated, and its sensory and motor mapping is hand-designed at the group level. Using it would put unverified engineering choices inside the "biological" part of the experiment. eonsystems confirms that the Shiu model can run on a GPU, but its GPL licence would bind this project. flyvis and FlyGym are not whole-brain simulators. flyvis comes back in §5 as a possible visual front end.

> **Selected:** the Shiu et al. (2024) LIF model on FlyWire v783, with its equations reproduced unchanged. Cell types come from the FlyWire annotation release (Schlegel et al. 2024, which includes the Matsliah et al. 2024 optic-lobe types). Exact file hashes and commits are in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

## 2. What exactly is simulated

```
dv/dt = (v_0 − v + g) / t_mbr        v_0 = −52 mV, t_mbr = 20 ms
dg/dt = −g / tau                     tau = 5 ms
v > −45 mV → spike; v = −52 mV, g = 0; refractory 2.2 ms
pre spike i → g_j += 0.275 mV × W[j,i]   after 1.8 ms
W[j,i] = (#synapses i→j) × (−1 if i is GABA/glutamatergic, else +1)
```

Every neuron uses the same parameters. The only neuron-specific biology is the **wiring and the transmitter sign**. This matters for evolution: thresholds, gains and synaptic scalings are the natural "knobs" (see §7).

## 3. Implementation: two backends, one model

* `brain/neural_simulator.py::brian2_reference_counts` runs Brian2 with the published equations. This is the ground truth.
* `brain/neural_simulator.py::TorchLIF` is a PyTorch reimplementation using a sparse CSR matrix and exact integration of the linear dynamics. It runs **B brains in parallel** as columns of a batch. Evolution needs this, because a whole population can then be evaluated in one GPU pass.

**Validation** (`experiments/validate_backends.py` → `results/milestone0/backend_validation.json`). Stimulus: left LPLC2 at 150 Hz, 200 ms, 4 trials per backend:

| metric | value |
|---|---|
| per-neuron firing-rate correlation (all active neurons) | **r = 0.966** |
| same, excluding the stimulated neurons | **r = 0.972** |
| Jaccard overlap of the set of neurons that fire | 0.76 |
| total spikes, Torch ÷ Brian2 | 1.11 |

The remaining differences come mostly from weakly active neurons, which fire stochastically. The 11% excess in total spikes is small but systematic enough to look into before the final experiments. It is logged in the DEVLOG as an open item. Unit tests (`tests/test_brain.py`, 7 passing) check the integration against the closed-form solution, the 1.8 ms delay, synaptic sign, Poisson drive rate and seed reproducibility.

## 4. Can it be stimulated and read? Yes, with one major caveat about the eye

### 4a. Visual population → descending neurons
`experiments/pathway_probe.py` → `results/milestone0/pathway_probe.json`, `dn_responses.csv`. Each visual population was driven at 150 Hz for 200 ms, one side at a time, with 3 repeats, while all **1,303 descending neurons** were recorded.

| stimulated (left side) | neurons | DNs responding > 5 Hz | strongest responders |
|---|---:|---:|---|
| R1-6 photoreceptors | 4,044 | **0** | none |
| R7/R8 photoreceptors | 1,330 | **0** | none |
| T4/T5 (direction-selective) | 6,146 | 94 | DNp26, DNg46, DNp31, DNp22 |
| LPLC2 (looming) | 108 | 76 | DNp103, DNp04, **DNp01 (giant fiber) 177 Hz** |
| LC4 (looming) | 54 | 39 | DNp04, DNp11, **DNp02** |
| LPLC1 (object/collision) | 68 | 110 | DNp03, DNp11, DNp06, **DNa01 (contra)** |
| LC9 | 106 | 93 | DNg16, DNg100, DNp09 |

The literature pathways are present and active without any tuning. LPLC2 and LC4 drive the giant fiber DNp01 (von Reyn et al. 2017; Ache et al. 2019). The steering neurons **DNa01/DNa02** (Rayshubskiy et al. 2020) respond mostly **contralaterally**. For example, right-side LC9 drives left DNa01 at 115 Hz and right DNa01 at 0 Hz. *Hypothesis for later testing, not a result:* if the decoder maps left-DNa01 to "turn left", the unmodified wiring biases the snake to turn away from objects on the right.

### 4b. Dose-response: the prototype for the distance-and-evasion analysis
Left LPLC2 input rate versus giant-fiber rate and latency (`B_C_…` in `pathway_probe.json`):

| LPLC2 input (Hz) | 0 | 10 | 25 | 50 | 75 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|---|
| DNp01-L rate (Hz) | 0 | 3 | 50 | 98 | 120 | 140 | 168 | 197 |
| first spike (ms) | – | 130 | 30 | 21 | 11 | 8 | 6 | 5.7 |

This is a graded, monotonic stimulus → neural response → latency curve. Section 11 of the brief asks for exactly this relationship, with "stimulus intensity" later replaced by "obstacle distance".

### 4c. The caveat: photoreceptor input does not propagate
`experiments/retina_propagation.py` → `results/milestone0/retina_propagation.json`. Driving all left R1-6 at 150 Hz gives L1 12 Hz, L2 18 Hz, L3 3 Hz, Tm1 2 Hz, and **0 Hz in Mi1, Tm3, T4, T5 and every lobula neuron**. Driving L1–L3 directly at 150 Hz also fails to activate T4/T5.

**Why** (this is a model limitation, not a data error):
1. Photoreceptors and lamina neurons use **graded potentials**, not spikes. The LIF model makes them spike.
2. Their signs are not fixed. Histamine from photoreceptors *hyperpolarises* L1/L2, and L1's glutamate acts through GluClα to make Mi1 an ON-pathway neuron. The Shiu convention (glutamate = inhibitory, everything non-GABA/Glu = excitatory) gets the first relay's sign wrong: R1-6 → L1 comes out as +35,845. It also makes L1 → Mi1 (−122,574) a plain inhibitory block.
3. Shiu et al. validated the model on gustatory, feeding and grooming circuits, **not on vision**.

**Consequence.** "Pixels → photoreceptors → whole connectome" is technically possible but biologically meaningless with this neural model: nothing would reach the central brain. §5 explains how the design handles this honestly.

## 5. Recommended visual interface (decision needed later; recommendation now)

| option | what is engineered | what is biological | verdict |
|---|---|---|---|
| **A. Photoreceptor entry** | pixel → R1-6/R7/R8 rate | everything downstream | Shown above to be dead in this LIF model. Keep it as a **control condition** only |
| **B. Feature-level entry** (recommended for Stage 4) | encoder converts the rendered eye view into rates for retinotopic **T4/T5** (local motion), **LPLC2/LC4** (looming), **LPLC1/LC9/LC11** (objects) | lobula → central brain → descending neurons → decoder | Pathways verified in §4. The encoder plays the role of the retina and medulla computations and is documented as **artificial** |
| **C. flyvis front end** (stretch goal) | mapping from flyvis's hex lattice to FlyWire T4/T5 columns | flyvis's connectome-constrained retina → T4/T5 + FlyWire brain | Most biological, but the column mapping between two datasets is research in its own right |

Under option B the encoder must not compute actions. It only reports retinotopic quantities that the fly's optic lobe is known to compute: angular expansion, local motion direction, and object presence per receptive field. Distance is never an input. Looming arises naturally as an obstacle's angular size grows while the snake approaches.

Retinotopic assignment of individual LC/T4/T5 neurons to eye positions is still to be done. FlyWire publishes column assignments (the Codex "visual columns" dataset). A fallback is to use synapse positions in the lobula. This is the first task of Stage 4.

## 6. Proposed motor readout (Stage 4, informed by §4a)

The action space has 3 options: left, straight, right. Candidate readout populations, all verified to exist and to respond to visual input:

* **turn**: DNa01 + DNa02 (left vs right). Their asymmetry votes left or right.
* **escape / evasive**: DNp01 (giant fiber), DNp02, DNp11, DNp04. Recorded for the evasion analysis. Whether they also drive the action is a configurable decoder choice.
* **reverse**: MDN (moonwalker). Recorded, but the action is disabled because Snake forbids reversing.

The decoder is a fixed, documented function of spike counts over a decision window. A typical rule: turn toward the side whose DNa population fires more, if the difference exceeds a margin; otherwise go straight. Its only free parameters are gains and margin, and those are evolvable genome entries. It contains no game logic.

## 7. Computational cost (the one-million-generation question)

`experiments/benchmark.py` → `results/milestone0/benchmark.json`, on the GTX 1660 Ti (6 GB):

| batch B | wall s per simulated s (total) | per brain | peak GPU |
|---:|---:|---:|---:|
| 1 | 16.1 | 16.1 | 0.2 GB |
| 16 | 124 | 7.8 | 0.4 GB |
| 64 | 219 | **3.4** | 1.2 GB |

The projection uses explicit, editable assumptions: 50 ms of brain time per Snake move, 500 moves per episode, 5 episodes per candidate, population 64. That gives **125 simulated brain-seconds per candidate**, or about **7.6 GPU-hours per generation**:

| generations | wall time on this PC |
|---:|---:|
| 1,000 | ~316 days |
| 10,000 | ~8.7 years |
| 1,000,000 | **~866 years** |

**Speedups that preserve the model**, to be measured, not assumed:
* **Event-driven synaptic update.** Profiling shows the dense-vector sparse multiply costs 15 ms per step at B=64, even though only ~0.5% of neurons spike in a step. Gathering only the spiking columns should give 10–50×.
* CUDA graphs or kernel fusion to remove Python per-step overhead (dominant at small B).
* A modern data-centre GPU: roughly 5–10× this card.
* Fewer brain-ms per decision (e.g. 20 ms). This is a scientific parameter and needs a sensitivity check.

Even an optimistic 1,000× total speedup puts 10⁶ generations near a year of compute on this card. Scope:

1. Treat **total evaluation budget** (simulated episodes), not "generations", as the x-axis. A generation's cost depends on population size, so "1 M generations" is not a fixed amount of optimisation.
2. Run pilot evolution at **10²–10⁴ generations** with checkpoints at 0 / 10 / 100 / 1,000 / 10,000. The 10⁵–10⁶ checkpoints are kept in the plan only if measured learning curves and post-optimisation throughput justify them.
3. The million-generation hypothesis is then tested by **extrapolation with confidence intervals** from several independent runs. The report will say plainly if it cannot be tested directly.

Any reduced model, such as pruning neurons more than k hops from both the sensory and the readout populations, would be a **separate, labelled experiment**. It never silently replaces the full connectome.

## 8. Recommended architecture (data flow)

```
 Python (authoritative)                                            Godot 4 (view only)
┌──────────────────────────────────────────────────────────────┐   ┌───────────────────┐
│ game/  SnakeState ─ step(action) ─ collisions ─ food (seeded) │──▶│ 3D arena, cameras │
│   │ grid state                                                │ WS│ overlays, replays │
│   ▼                                                           │◀──│ human input       │
│ brain/visual_encoder   eye-view raycast → retinotopic features │   └───────────────────┘
│   │ Poisson rates (Hz) for T4/T5, LPLC2, LC4, LPLC1…           │
│   ▼                                                           │
│ brain/neural_simulator TorchLIF (138,639 neurons, B brains)    │
│   │ spike counts of DNa01/02, DNp01/02/11, MDN …               │
│   ▼                                                           │
│ brain/motor_decoder    counts → {left, straight, right}        │
│ brain/neural_recording per-decision logs (distance, rates…)    │
│ evolution/             genome = gains/thresholds/scalings      │
│ agents/                A* (original), random, reactive, MLP    │
│ experiments/           benchmark, crossover, self-trapping …   │
└──────────────────────────────────────────────────────────────┘
```

* The **game runs headless** in Python for training. Godot only renders: it receives state over a local WebSocket, or replays a recorded JSONL episode. Training never waits on rendering, and the simulation stays deterministic.
* For training, the **virtual eye** is a cheap grid raycast in Python ("what occupies each direction of the visual field, at what angular size"). That is the information the Godot snake-perspective camera would show. Godot's camera view is for visual checking and the visual-perspective generalisation test, not for the training loop.

## 9. Installation requirements

| component | version used / required |
|---|---|
| Python | 3.11 |
| PyTorch | 2.5.1 + CUDA 12.1 wheels (`--index-url https://download.pytorch.org/whl/cu121`) |
| Brian2 | 2.9.0 (reference backend only). **Requires numpy < 2.3** (2.4 removed `ndarray.ptp`) |
| numpy / pandas / pyarrow / scipy | 2.2 / 2.3 / 26 / 1.17 |
| GPU | any CUDA GPU with ≥ 2 GB of free memory. CPU works but is slow |
| Godot | 4.x (Stage 1; not needed for Milestone 0) |
| Disk | ~140 MB of data (`scripts/fetch_data.py`) |

Windows note: put the virtual environment on a **short path** (e.g. `C:\Users\<you>\.venvs\flysnake3d`). With a long path, pyarrow's DLLs fail to load ("filename or extension is too long").

## 10. What this milestone does *not* establish

* That any of these pathways *mean* anything in Snake. Stimulus → DN response has been shown. Behaviour has not.
* That the LIF model is realistic for visual neurons. §4c shows that it is not for the retina and lamina.
* Retinotopy of individual LC/T4/T5 neurons. This is still to be done.

## 11. Next step

**Milestone 1 = Stage 1 (headless Python Snake + Godot viewer)**, built test-first:
1. `game/` deterministic grid Snake: seeded food, wall and self collision, no reversing, configurable board size, and a JSONL replay format.
2. `agents/astar_agent.py` reproducing the user's original A* *exactly*. **This needs the user's existing A* code.** It must not be re-invented, so that the comparison targets the real baseline.
3. A Godot 4 project that renders a live or replayed game, with the four camera modes.

Stage 4 (encoder/decoder) then connects this game to the brain verified here.

## References
Shiu P.K. et al. (2024) *Nature* 634:210–219 (connectome LIF model) · Dorkenwald S. et al. (2024) *Nature* 634:124–138 (FlyWire) · Schlegel P. et al. (2024) *Nature* 634:139–152 (annotations) · Matsliah A. et al. (2024) *Nature* (optic-lobe cell types) · von Reyn C.R. et al. (2017) *Curr Biol* (LC4 → giant fiber) · Ache J.M. et al. (2019) *Curr Biol* (LPLC2 → giant fiber) · Rayshubskiy A. et al. (2020) bioRxiv (DNa01/DNa02 steering) · Bidaye S.S. et al. (2014) *Science* (MDN) · Lappalainen J.K. et al. (2024) *Nature* (flyvis) · Wang-Chen S. et al. (2024) *Nat Methods* (NeuroMechFly v2).
*Page numbers are given only where verified; check the rest against the publisher before citing.*
