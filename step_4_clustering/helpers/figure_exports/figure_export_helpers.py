from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def save_current_figure(output_dir: Path, name: str, formats: Iterable[str] = ("pdf", "png"), dpi: int = 220) -> None:
    """Save the current plot in the requested formats."""
    for ext in formats:
        plt.savefig(output_dir / f"{name}.{ext}", bbox_inches="tight", dpi=dpi)
    plt.close()
