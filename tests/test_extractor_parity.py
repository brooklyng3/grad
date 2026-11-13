"""Parity test: compiled AfterImage must match the pure-Python extractor."""

import os
import subprocess
import sys

import numpy as np

FEATURE_SCRIPT = """
import sys
import numpy as np
sys.path.insert(0, "Kitsune-py")
from FeatureExtractor import FE

limit = 200
fe = FE("data/raw/mirai.pcap.tsv", limit=limit)
rows = []
while len(rows) < limit:
    vector = fe.get_next_vector()
    if len(vector) == 0:
        break
    rows.append(vector)
np.save(sys.argv[1], np.asarray(rows, dtype=np.float32))
"""


def _extract(path: str, use_cython: bool) -> np.ndarray:
    env = dict(os.environ)
    if use_cython:
        env["KITSUNE_USE_CYTHON"] = "1"
    subprocess.run(
        [sys.executable, "-c", FEATURE_SCRIPT, path],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return np.load(path)


def test_cython_matches_pure_python(tmp_path):
    pure = _extract(str(tmp_path / "pure.npy"), use_cython=False)
    compiled = _extract(str(tmp_path / "cython.npy"), use_cython=True)

    assert pure.shape == compiled.shape == (200, 100)
    assert np.isfinite(compiled).all()
    assert float(np.max(np.abs(pure - compiled))) < 1e-6
