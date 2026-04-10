from scipy.optimize import linear_sum_assignment
from rhythm import *
import pandas as pd
from scipy.stats import t as student_t


def _xy_set(r: Rhythm) -> Set[Tuple[float, float]]:
    x, y = r.events()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Explicit 2-tuple of floats => Set[Tuple[float, float]]
    return {(float(xi), float(yi)) for xi, yi in zip(x, y)}

def pair_f1(a: Rhythm, b: Rhythm) -> float:
    A, B = _xy_set(a), _xy_set(b)
    denom = len(A) + len(B)
    if denom == 0:
        return 1.0
    inter = len(A & B)
    return 2.0 * inter / denom

def pair_jaccard(a: Rhythm, b: Rhythm) -> float:
    A, B = _xy_set(a), _xy_set(b)
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / float(len(A | B))

def pair_overlap(a: Rhythm, b: Rhythm) -> float:
    A, B = _xy_set(a), _xy_set(b)
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / float(min(len(A), len(B)))

# --- metric-agnostic matrix (always (x, y)) ---

def pairwise_matrix(
    true_rhythms: Iterable[Rhythm],
    detected_rhythms: Iterable[Rhythm],
    metric: Union[str, Callable[[Rhythm, Rhythm], float]] = "jaccard",
) -> pd.DataFrame:
    tr = list(true_rhythms)
    dr = list(detected_rhythms)

    if isinstance(metric, str):
        m = metric.lower()
        if m == "f1":
            pair_fn = pair_f1
        elif m == "jaccard":
            pair_fn = pair_jaccard
        elif m in ("overlap", "overlap_coefficient", "ss"):
            pair_fn = pair_overlap
        else:
            raise ValueError("metric must be 'f1', 'jaccard', 'overlap', or a callable")
    else:
        pair_fn = metric

    M = np.zeros((len(tr), len(dr)), float)
    for i, rt in enumerate(tr):
        for j, rd in enumerate(dr):
            M[i, j] = pair_fn(rt, rd)

    return pd.DataFrame(
        M,
        index=[r.name or f"R{i}" for i, r in enumerate(tr)],
        columns=[r.name or f"D{j}" for j, r in enumerate(dr)],
    )


def matching_score(M: np.ndarray, normalize: str = "max") -> float:
    """
    Overall match quality from a pairwise score matrix M (rows=true, cols=detected).
    Uses Hungarian assignment to MAXIMIZE total matched score, then normalizes.

    normalize:
      - "max"      -> divide by max(n_true, n_detected)   (symmetric default)
      - "true"     -> divide by n_true                    (recall-like)
      - "detected" -> divide by n_detected                (precision-like)
    """
    M = np.asarray(M, dtype=float)
    n_true, n_det = M.shape
    if n_true == 0 and n_det == 0:
        return 1.0

    # Pad to square with zeros so we can assign 1-to-1
    N = max(n_true, n_det)
    P = np.zeros((N, N), dtype=float)
    P[:n_true, :n_det] = M

    # Hungarian solves a min-cost problem; negate to maximize
    rows, cols = linear_sum_assignment(-P)
    diag_sum = float(P[rows, cols].sum())

    if normalize == "true":
        denom = max(1, n_true)
    elif normalize == "detected":
        denom = max(1, n_det)
    else:  # "max"
        denom = max(1, N)

    return diag_sum / denom



from collections import defaultdict
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def _blend_colors(colors, alpha=1.0):
    """
    Blend a list of matplotlib colors by averaging RGB channels.
    Returns an RGBA tuple.
    """
    if len(colors) == 0:
        return mcolors.to_rgba("C0", alpha=alpha)
    rgba = np.array([mcolors.to_rgba(c) for c in colors], dtype=float)
    rgb_mean = rgba[:, :3].mean(axis=0)
    return (*rgb_mean, alpha)


def plot_pulse_signal(
    x: np.ndarray,
    y: np.ndarray,
    run_length: float,
    radius: float = 0.01,
    title: str = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    mark_peaks: bool = True,
    show_legend: bool = True,
    pulse_facecolors=None,   # NEW
    pulse_edgecolors=None,   # NEW
    peak_colors=None,        # NEW
) -> plt.Axes:
    """
    Draw rectangular pulses *at the actual x locations*:
      for each (x[i], y[i]), fill the rectangle [x[i]-r, x[i]+r] × [0, y[i]].

    Optional per-pulse styling:
      - pulse_facecolors[i]
      - pulse_edgecolors[i]
      - peak_colors[i]
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    created = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created = True

    if x.size == 0:
        ax.set_title(title)
        if show and created:
            plt.tight_layout()
            plt.show()
        return ax

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # Reorder color arrays to match sorted x
    if pulse_facecolors is None:
        pulse_facecolors = ["C0"] * len(x)
    else:
        pulse_facecolors = np.asarray(pulse_facecolors, dtype=object)[order]

    if pulse_edgecolors is None:
        pulse_edgecolors = pulse_facecolors
    else:
        pulse_edgecolors = np.asarray(pulse_edgecolors, dtype=object)[order]

    if peak_colors is None:
        peak_colors = ["tab:orange"] * len(x)
    else:
        peak_colors = np.asarray(peak_colors, dtype=object)[order]

    for xi, yi, fc, ec in zip(x, y, pulse_facecolors, pulse_edgecolors):
        xl, xr = xi - radius, xi + radius
        ax.fill_between(
            [xl, xr], [yi, yi], [0.0, 0.0],
            color=fc, alpha=0.30, zorder=1
        )
        ax.plot(
            [xl, xl, xr, xr, xl], [0.0, yi, yi, 0.0, 0.0],
            color=ec, linewidth=1.2, zorder=2
        )

    if mark_peaks:
        ax.scatter(
            x, y, marker="x", s=70, c=list(peak_colors),
            zorder=3, label="Peak"
        )

    ax.set_xlim(-0.01 * run_length, run_length * 1.01)
    y_max = float(y.max()) if y.size else 1.0
    ax.set_ylim(0.0, y_max * 1.25 if y_max > 0 else 1.0)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            uniq = dict(zip(labels, handles))
            legend = ax.legend(uniq.values(), uniq.keys(), loc="upper right", frameon=True)
            plt.draw()
            bbox = legend.get_window_extent(ax.figure.canvas.get_renderer())
            inv = ax.transAxes.inverted()
            bbox_axes = inv.transform(bbox)
            legend_height = bbox_axes[1, 1] - bbox_axes[0, 1]
            if legend_height > 0:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, ymax * (1 + legend_height * 1.1))

    if show and created:
        plt.tight_layout()
        plt.show()

    return ax


def plot_rhythm_peak_sets(
    x,
    y,
    rhythms,
    run_length,
    radius=0.01,
    match_tol=None,
    title=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from collections import defaultdict

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape

    if match_tol is None:
        match_tol = max(radius * 0.5, 1e-9)

    cmap = plt.colormaps["Set1"]

    def _blend_colors(colors, alpha=1.0):
        rgba = np.array([mcolors.to_rgba(c) for c in colors], dtype=float)
        rgb_mean = rgba[:, :3].mean(axis=0)
        return (*rgb_mean, alpha)

    # --- Sort x once for nearest-time matching ---
    order = np.argsort(x)
    x_sorted = x[order]

    def times_to_original_indices(t_arr, x_sorted, order, tol):
        t_arr = np.asarray(t_arr, dtype=float)
        pos = np.searchsorted(x_sorted, t_arr)

        left = np.clip(pos - 1, 0, len(x_sorted) - 1)
        right = np.clip(pos, 0, len(x_sorted) - 1)

        dleft = np.abs(t_arr - x_sorted[left])
        dright = np.abs(t_arr - x_sorted[right])
        nearest_sorted = np.where(dleft <= dright, left, right)

        hits = []
        for i_t, i_sorted in enumerate(nearest_sorted):
            if abs(t_arr[i_t] - x_sorted[i_sorted]) <= tol:
                hits.append(int(order[i_sorted]))
        return sorted(set(hits))

    # --- Map each rhythm to its displayed candidate number ---
    # Uses r.candidate_number if present; otherwise falls back to list position.
    candidate_numbers = []
    for ri, r in enumerate(rhythms):
        cand_num = getattr(r, "candidate_number", None)
        if cand_num is None:
            cand_num = ri
        candidate_numbers.append(cand_num)

    # --- peak_to_users[i] = list of rhythm indices using peak i ---
    peak_to_users = defaultdict(list)

    for ri, r in enumerate(rhythms):
        r_times = getattr(r, "x", r)
        idxs = times_to_original_indices(r_times, x_sorted, order, match_tol)
        for idx in idxs:
            peak_to_users[idx].append(ri)

    # --- Assign a color to each unique membership pattern ---
    combo_to_color = {}
    for users in peak_to_users.values():
        combo = tuple(sorted(set(users)))
        if combo not in combo_to_color:
            if len(combo) == 1:
                combo_to_color[combo] = cmap(combo[0] % cmap.N)
            else:
                combo_to_color[combo] = _blend_colors(
                    [cmap(u % cmap.N) for u in combo],
                    alpha=1.0,
                )

    # --- Per-peak colors ---
    pulse_facecolors = []
    pulse_edgecolors = []
    peak_colors = []

    for i in range(len(x)):
        combo = tuple(sorted(set(peak_to_users.get(i, []))))
        if len(combo) == 0:
            color = "lightgray"
            edge = "gray"
        else:
            color = combo_to_color[combo]
            edge = color

        pulse_facecolors.append(color)
        pulse_edgecolors.append(edge)
        peak_colors.append(edge)

    # --- Plot signal ---
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pulse_signal(
        x,
        y,
        run_length,
        radius=radius,
        title=title,
        ax=ax,
        show=False,
        mark_peaks=True,
        show_legend=False,
        pulse_facecolors=pulse_facecolors,
        pulse_edgecolors=pulse_edgecolors,
        peak_colors=peak_colors,
    )

    # --- Build generalized legend ---
    legend_handles = []
    legend_labels = []

    def combo_label(combo):
        pieces = [rf"S_{{{candidate_numbers[i]}}}" for i in combo]
        return r"$" + r" \cap ".join(pieces) + r"$"
    #     # Hardcode: index 0 -> candidate 1, index 1 -> candidate 10
    #     mapping = {0: 1, 1: 10}
    #     pieces = [rf"S_{{{mapping[i]}}}" for i in combo if i in mapping]
    #     return r"$" + r" \cap ".join(pieces) + r"$"

    # Sort combos so singletons come first, then intersections
    combos_present = sorted(
        combo_to_color.keys(),
        key=lambda c: (len(c), tuple(candidate_numbers[i] for i in c))
    )

    for combo in combos_present:
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                marker="x",
                linestyle="",
                color=combo_to_color[combo],
                markersize=8,
            )
        )
        legend_labels.append(combo_label(combo))

    if any(len(peak_to_users.get(i, [])) == 0 for i in range(len(x))):
        legend_handles.append(
            plt.Line2D(
                [],
                [],
                marker="x",
                linestyle="",
                color="gray",
                markersize=8,
            )
        )
        legend_labels.append("Unassigned")

    if legend_handles:
        ax.legend(legend_handles, legend_labels, loc="upper right")

    plt.tight_layout()
    plt.show()


def plot_scores(
    score_groups: dict[str, dict[float, list]],
    title: str | None = None,
    confidence: float = 0.95,
    smoothing: int | None = None,
    vline_x: float | None = 0.1,
    xlabel: str = "σₓ",
    ylabel: str = "Jaccard Matching Score (JMS)",
    ylim: tuple[float, float] | None = (-0.05, 1.05),
    colors: list[str] | None = None,
):
    """
    Plot multiple score groups with mean +/- CI shading (smoothed), optional smoothing,
    optional dashed vertical reference line, and CI bounds clamped to [0, 1].

    If `colors` is provided, it should be a list of matplotlib-compatible color
    specs; the i-th score_group will use colors[i]. If `colors` is None, the
    default color cycle is used (as before).
    """
    alpha2 = (1.0 - float(confidence)) / 2.0

    fig, ax = plt.subplots(figsize=(9, 4.5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not color_cycle:
        color_cycle = [f"C{i}" for i in range(10)]

    # Centered moving average with edge padding
    def smooth(arr, k):
        if k <= 1:
            return arr
        k = min(k, len(arr))
        kernel = np.ones(k) / k
        pad = k // 2
        arr_padded = np.pad(arr, (pad, pad), mode="edge")
        return np.convolve(arr_padded, kernel, mode="valid")

    for i, (label, series) in enumerate(score_groups.items()):
        xs = sorted(series.keys())

        means = []
        halfw = []

        for x in xs:
            vals = np.asarray(series[x], dtype=float)
            vals = vals[np.isfinite(vals)]
            n = len(vals)
            if n == 0:
                means.append(np.nan)
                halfw.append(np.nan)
                continue

            m = float(np.mean(vals))
            if n > 1:
                s = float(np.std(vals, ddof=1))
                se = s / np.sqrt(n)
                tcrit = float(student_t.ppf(1.0 - alpha2, df=n - 1))
                hw = tcrit * se
            else:
                hw = 0.0

            means.append(m)
            halfw.append(hw)

        xs = np.asarray(xs, float)
        means = np.asarray(means, float)
        halfw = np.asarray(halfw, float)

        # pick color: user-specified list (if given) or fall back to cycle
        if colors is not None and i < len(colors):
            color = colors[i]
        else:
            color = color_cycle[i % len(color_cycle)]

        k = max(1, int(smoothing)) if smoothing else 1
        means_s = smooth(means, k)
        halfw_s = smooth(halfw, k)

        # Reconstruct bounds
        lower_s = means_s - halfw_s
        upper_s = means_s + halfw_s

        # Clamp to [0, 1]
        lower_s = np.clip(lower_s, 0.0, 1.0)
        upper_s = np.clip(upper_s, 0.0, 1.0)

        # CI shading
        ax.fill_between(xs, lower_s, upper_s, color=color, alpha=0.4, linewidth=0)

        # smoothed mean line
        ax.plot(xs, means_s, color=color, linewidth=1, label=label)

        # raw means as points
        ax.scatter(
            xs,
            means,
            color=color,
            s=20,
            marker="o",
            edgecolors="white",
            alpha=0.8,
            linewidths=0.7,
            zorder=5,
        )

    # vertical reference
    if vline_x is not None:
        ax.axvline(vline_x, linestyle="--", color="black", alpha=0.55, linewidth=0.5)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.4)

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = True
            uniq_handles.append(h)
            uniq_labels.append(l)
    ax.legend(uniq_handles, uniq_labels)

    fig.tight_layout()
    plt.show()
