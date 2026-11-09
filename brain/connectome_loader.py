"""Load the FlyWire v783 connectome exactly as the Shiu et al. (2024) model uses it.

Data model
----------
* Neurons are addressed by a dense *simulator index* 0..N-1. The index order is
  the row order of Completeness_783.csv (identical to the Shiu Brian2 model), so
  results here can be compared neuron-for-neuron with the reference model.
* Each neuron also has its FlyWire ``root_id`` (an int64 like 720575940596125868)
  and, where available, annotations: ``cell_type``, ``super_class``, ``side``.
* Edges are stored as a signed synapse-count matrix ``W[post, pre]``:
      W[j, i] = n_synapses(i -> j) * sign(i)
  where sign(i) = -1 if neuron i's predicted transmitter is GABA or glutamate,
  else +1 (Shiu et al.'s convention). This matrix is the *biological reference*;
  nothing in this module modifies it. Evolution applies changes on top of a copy.

Nothing here invents a neuron or a pathway: every cell-type query goes through
the published annotation table, and an unknown type raises an error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class Connectome:
    root_ids: np.ndarray                 # (N,) int64 FlyWire IDs, index order
    annotations: pd.DataFrame            # indexed by simulator index
    W: sp.csr_matrix                     # (N, N) signed synapse counts, W[post, pre]
    source: dict = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return len(self.root_ids)

    @property
    def n_edges(self) -> int:
        return self.W.nnz

    # ---- queries -------------------------------------------------------
    def indices_of_type(self, cell_type: str, side: str | None = None) -> np.ndarray:
        """Simulator indices of all neurons annotated with ``cell_type``.

        Matches either FlyWire ``cell_type`` or ``hemibrain_type``. Raises
        KeyError if no neuron has that type, so a typo can never silently
        produce an empty stimulus.
        """
        a = self.annotations
        m = (a.cell_type == cell_type) | (a.hemibrain_type == cell_type)
        if side is not None:
            m &= a.side == side
        idx = np.flatnonzero(m.to_numpy())
        if idx.size == 0:
            raise KeyError(f"no neurons of type {cell_type!r} (side={side}) in connectome")
        return idx

    def indices_of_super_class(self, super_class: str, side: str | None = None) -> np.ndarray:
        a = self.annotations
        m = a.super_class == super_class
        if side is not None:
            m &= a.side == side
        return np.flatnonzero(m.to_numpy())

    def synapses_between(self, pre_type: str, post_type: str) -> int:
        """Signed synapse count summed over all (pre_type -> post_type) pairs."""
        pre = self.indices_of_type(pre_type)
        post = self.indices_of_type(post_type)
        return int(self.W[post][:, pre].sum())


def load_connectome(data_dir: Path = DATA_DIR) -> Connectome:
    comp = pd.read_csv(data_dir / "Completeness_783.csv", index_col=0)
    root_ids = comp.index.to_numpy(dtype=np.int64)
    n = len(root_ids)

    con = pd.read_parquet(
        data_dir / "Connectivity_783.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = con["Presynaptic_Index"].to_numpy(np.int64)
    post = con["Postsynaptic_Index"].to_numpy(np.int64)
    w = con["Excitatory x Connectivity"].to_numpy(np.float32)
    # The parquet has one row per (pre, post) pair; coo->csr sums duplicates anyway.
    W = sp.coo_matrix((w, (post, pre)), shape=(n, n)).tocsr()

    ann = pd.read_csv(
        data_dir / "neuron_annotations.tsv",
        sep="\t",
        usecols=["root_id", "super_class", "cell_class", "cell_type", "hemibrain_type", "side", "top_nt"],
        dtype={"root_id": np.int64},
        low_memory=False,
    ).drop_duplicates("root_id").set_index("root_id")
    ann = ann.reindex(root_ids).reset_index().rename(columns={"index": "root_id"})

    return Connectome(
        root_ids=root_ids,
        annotations=ann,
        W=W,
        source={
            "connectome": "FlyWire FAFB v783 (Dorkenwald et al. 2024)",
            "model_files": "philshiu/Drosophila_brain_model (Shiu et al. 2024)",
            "annotations": "flyconnectome/flywire_annotations (Schlegel et al. 2024)",
            "n_neurons": n,
            "n_edges": int(W.nnz),
            "n_unannotated": int(ann.cell_type.isna().sum()),
        },
    )
