"""Shared helpers: 16-bit -> 8-bit scaling used by both the MP4 recorder and the live stream."""

from typing import Optional

import numpy as np


class AutoGain:
    """Percentile stretch with exponential smoothing so the picture doesn't flicker."""

    ALPHA = 0.15

    def __init__(self):
        self.lo: Optional[float] = None
        self.hi: Optional[float] = None

    def bounds(self, frame: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(frame[::4, ::4], (1, 99))
        if self.lo is None:
            self.lo, self.hi = float(lo), float(hi)
        else:
            self.lo += self.ALPHA * (lo - self.lo)
            self.hi += self.ALPHA * (hi - self.hi)
        return self.lo, max(self.hi, self.lo + 1.0)


def to_8bit(frame: np.ndarray, agc: AutoGain) -> np.ndarray:
    """Scale a 2-D uint16 frame to uint8 using the (smoothed) auto-gain bounds."""
    lo, hi = agc.bounds(frame)
    return np.clip((frame.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
