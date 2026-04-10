from typing import Set, Tuple, Iterable, Callable, Union, List, Optional
import numpy as np

class Rhythm:
    """
    Minimal container for a rhythm's events.
    Holds sorted x and y.
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        period: Optional[float] = None,
        run_length: Optional[float] = None,
        name: Optional[str] = None,
        prefix: Optional[str] = "T"
    ) -> None:
        if name is None:
            name = f"{prefix}-{period:g}"
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size != y.size:
            raise ValueError("x and y must have the same length")

        order = np.argsort(x)
        self.x = x[order]
        self.y = y[order]
        self.period = period
        self.run_length = run_length
        self.name = name

    def __len__(self) -> int:
        return int(self.x.size)

    def events(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x, y) event arrays (sorted by x)."""
        return self.x, self.y

    def to_set_exact(self) -> set:
        """Exact-match set of event times (deduped)."""
        return set(np.unique(self.x))

    def to_set_xy(self):
        return set(zip(self.x.tolist(), self.y.tolist()))

    def __repr__(self) -> str:
        label = self.name if self.name is not None else "Rhythm"
        extras = []
        if self.period is not None:
            extras.append(f"period={self.period}")
        if self.run_length is not None:
            extras.append(f"run_length={self.run_length}")
        extras.append(f"points={len(self)}")
        return f"{label}(" + ", ".join(extras) + ")"


def make_rhythm(
    period: float,
    height: float,
    run_length: float,
    x_sd: float = 0.0,
    y_sd: float = 0.0,
    seed: Optional[int] = None,
    name: Optional[str] = None,
) -> Rhythm:
    """
    Generate a synthetic rhythm and return a Rhythm object.
    Always clips x into [0, run_length] and clamps y >= 0.
    """
    period = float(period)
    height = float(height)
    run_length = float(run_length)
    if not (0.0 < period < run_length):
        raise ValueError("period must satisfy 0 < period < run_length")
    if x_sd < 0 or y_sd < 0:
        raise ValueError("x_sd and y_sd must be >= 0")

    rng = np.random.default_rng(seed)

    n = int(np.floor(run_length / period))
    if n <= 0:
        return Rhythm(np.array([]), np.array([]), period=period, run_length=run_length, name=name)

    x_nom = period * np.arange(1, n + 1, dtype=float)
    x = x_nom + (rng.normal(0.0, x_sd, size=n) if x_sd > 0 else 0.0)
    y = rng.normal(height, y_sd, size=n) if y_sd > 0 else np.full(n, height, dtype=float)

    x = np.clip(x, 0.0, run_length)
    y = np.maximum(y, 0.0)

    return Rhythm(x, y, period=period, run_length=run_length, name=name)


def compose_rhythms(
        rhythms: Iterable["Rhythm"],
        dedup: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge multiple rhythms into global (x, y), sorted by x.

    Parameters
    ----------
    rhythms : Iterable[Rhythm]
        List of Rhythm objects providing .events() -> (x, y).
    dedup : bool, default=True
        If True, remove duplicate x-values after sorting (keep first occurrence).
    """
    xs, ys = [], []
    for r in rhythms:
        if len(r):
            x_i, y_i = r.events()
            xs.append(x_i)
            ys.append(y_i)
    if not xs:
        return np.array([]), np.array([])

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    order = np.argsort(x)
    x, y = x[order], y[order]

    if dedup:
        mask = np.concatenate(([True], np.diff(x) != 0))
        x, y = x[mask], y[mask]

    return x, y