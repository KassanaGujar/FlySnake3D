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
4. **Need the user's original A* code** for the baseline (Stage 2).
5. Godot 4 is not installed on the dev machine yet.

**Next:** Milestone 1, a headless Snake core with tests and a replay format, followed by the Godot viewer.
