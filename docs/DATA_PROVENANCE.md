# Data provenance

All biological data is downloaded by `scripts/fetch_data.py`, pinned to an exact upstream commit and checked by SHA-256. None of it is committed to this repository. One reason is size: `Connectivity_783.parquet` is above GitHub's 100 MB per-file limit. The other is that the original authors remain the single source of truth.

| file | source | commit | SHA-256 | licence |
|---|---|---|---|---|
| `Completeness_783.csv` | philshiu/Drosophila_brain_model | `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960` | `bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311` | MIT |
| `Connectivity_783.parquet` | philshiu/Drosophila_brain_model | same | `efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347` | MIT |
| `neuron_annotations.tsv` (= `supplemental_files/Supplemental_file1_neuron_annotations.tsv`) | flyconnectome/flywire_annotations | `a83b2776d60d5764cef36b927f5f9679c16c47a2` | `b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd` | see upstream repo (no LICENSE file found at that commit; data accompanies Schlegel et al. 2024, *Nature*) |

Note: a `git clone` on Windows can rewrite line endings in the CSV, which changes its hash. Use the raw download.

## Contents as verified (by `tests/test_brain.py::test_real_connectome_loads` and the probes)

* 138,639 neurons. Index order = row order of `Completeness_783.csv` = Shiu Brian2 index.
* 15,091,983 directed neuron→neuron connections carrying 54,492,922 synapses. 9,059,302 connections are excitatory (+1) and 6,032,681 inhibitory (−1).
* Sign rule (Shiu et al.): presynaptic GABA or glutamate gives −1; any other predicted transmitter gives +1.
* Annotations cover 138,625 of the model's neurons. 14 model IDs are missing from the annotation table, and 1,492 neurons have no `cell_type`.
* Super classes: optic 77,541 · central 32,383 · sensory 16,907 · visual_projection 8,038 · ascending 1,750 · descending 1,303 · visual_centrifugal 524 · motor 110 · endocrine 80 (annotation table rows).

## Citations
* Dorkenwald S. et al. (2024) Neuronal wiring diagram of an adult brain. *Nature* 634:124–138.
* Schlegel P. et al. (2024) Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634:139–152.
* Shiu P.K. et al. (2024) A *Drosophila* computational brain model reveals sensorimotor processing. *Nature* 634:210–219.
* FlyWire data is released under the terms on https://codex.flywire.ai. Cite the above when using it.
