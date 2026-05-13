from __future__ import annotations

import json
import os
import pickle as pkl
from typing import Dict


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_results(results: Dict, out_dir: str) -> None:
    ensure_dir(out_dir)
    # JSON
    with open(os.path.join(out_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # PKL
    with open(os.path.join(out_dir, "test_results.pkl"), "wb") as f:
        pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)

