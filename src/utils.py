"""Small helpers shared by the pipeline modules (JSON I/O, logging, figure style)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display needed; works on servers and Windows alike
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src import config


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints timestamps (configured once in run_all.py)."""
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%H:%M:%S")
    return logging.getLogger(name)


class _NumpyEncoder(json.JSONEncoder):
    """Lets json.dump handle numpy numbers/arrays."""

    def default(self, o: Any):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, cls=_NumpyEncoder)


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def update_json_section(path: Path, section: str, value: Any) -> None:
    """Replace one top-level key of a JSON file, keeping the other keys."""
    data = load_json(path)
    data[section] = value
    save_json(data, path)


def update_metrics(section: str, value: Any) -> None:
    update_json_section(config.METRICS_JSON, section, value)


def set_plot_style() -> None:
    """One consistent, paper-friendly style for every matplotlib figure."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": config.FIG_DPI,
        "savefig.bbox": "tight",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def savefig(fig, name: str) -> Path:
    """Save a figure as 300-dpi PNG in reports/figures and close it."""
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


CLASS_PALETTE = {0: config.COLOR_NOT_PLACED, 1: config.COLOR_PLACED}
CLASS_NAMES = {0: "Not Placed", 1: "Placed"}
