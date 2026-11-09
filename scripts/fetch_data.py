"""Download the connectome data files used by this project.

Why a download script instead of committing the data?
  * Connectivity_783.parquet is ~101 MB, above GitHub's 100 MB per-file limit.
  * Pinning the exact upstream commit + SHA-256 makes the experiment reproducible:
    anyone can verify they simulate the *same* wiring diagram we did.

Sources (all public):
  1. philshiu/Drosophila_brain_model  (MIT) - Shiu et al. 2024, Nature 634:210-219.
     Completeness_783.csv      : ordered list of the 138,639 FlyWire v783 neuron IDs
                                 (row order == simulator index).
     Connectivity_783.parquet  : 15,091,983 directed neuron->neuron edges with
                                 synapse counts and a +1/-1 sign derived from the
                                 predicted neurotransmitter.
  2. flyconnectome/flywire_annotations - Schlegel et al. 2024, Nature 634:139-152
     (includes Matsliah et al. 2024 optic-lobe types).
     Supplemental_file1_neuron_annotations.tsv : cell types, super classes, side.

Usage:  python scripts/fetch_data.py            (downloads into ./data)
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

SHIU_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
ANNOT_COMMIT = "a83b2776d60d5764cef36b927f5f9679c16c47a2"

FILES = {
    "Completeness_783.csv": (
        f"https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/{SHIU_COMMIT}/Completeness_783.csv",
        None,
    ),
    "Connectivity_783.parquet": (
        f"https://github.com/philshiu/Drosophila_brain_model/raw/{SHIU_COMMIT}/Connectivity_783.parquet",
        None,
    ),
    "neuron_annotations.tsv": (
        f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOT_COMMIT}/"
        "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        None,
    ),
}

# Filled in from the files we actually used (see docs/DATA_PROVENANCE.md).
EXPECTED_SHA256 = {
    "Completeness_783.csv": "bbb847a4cc2caaa7a16349722d220c087317b946d148d4d592d94d250617a311",
    "Connectivity_783.parquet": "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
    "neuron_annotations.tsv": "b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(dest: Path = Path(__file__).resolve().parents[1] / "data") -> int:
    dest.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, (url, _) in FILES.items():
        out = dest / name
        if not out.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, out)
        digest = sha256(out)
        expected = EXPECTED_SHA256.get(name)
        status = "unchecked" if not expected else ("OK" if digest == expected else "MISMATCH")
        ok &= status != "MISMATCH"
        print(f"{name:32s} {out.stat().st_size/1e6:8.1f} MB  sha256={digest[:16]}...  {status}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
