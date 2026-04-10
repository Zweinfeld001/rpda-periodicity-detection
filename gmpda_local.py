from gmpda import GMPDA
from scipy.stats import norm
from rhythm import *
from util import *
import numpy as np


def peaks_to_ts(x, run_length, sr=100):
    """ Convert peak times x (same units as run_length) into a binary 1D series ts
    sampled at 'sr' samples per unit. Returns ts and sr. """
    x = np.asarray(x, float)
    n = int(np.ceil(run_length * sr)) + 1
    ts = np.zeros(n, dtype=int)
    idx = np.clip(np.round(x * sr).astype(int), 0, n-1)
    ts[idx] = 1
    return ts, sr

def rhythm_from_mu(
    x: np.ndarray,
    y: np.ndarray,
    mu: float,
    sigma: float,
    run_length: float,
    confidence: float = 0.95,
) -> Rhythm:
    """
    Recover a Rhythm for a given candidate period mu by sweeping phase.

    We:
      - infer a tolerance window spread = z * sigma,
        where z is the z-score for the requested confidence level
        on a symmetric +/- interval (≈1.96 for 95%)
      - try different phase offsets φ in [0, mu)
      - lay down grid ticks: φ, φ+mu, φ+2mu, ...
      - match real peaks whose |x_peak - tick| <= spread
      - keep the phase that explains the most peaks
      - return a Rhythm with those matched peaks

    Parameters
    ----------
    x : array-like of float
        Observed peak times, same units as mu (e.g. seconds).
    y : array-like of float
        Peak heights at those times (same length as x). Used only to
        attach amplitudes to the returned Rhythm.
    mu : float
        Candidate period (same units as x).
    sigma : float
        Estimated jitter scale for this period, in the same units as x.
        (Typically sigma_samples / sr.)
    run_length : float
        Total duration of the region, same units as x.
    name : str, optional
        Name to assign to the resulting Rhythm (e.g. "G0").
    confidence : float
        Desired two-sided confidence mass for |error| <= spread.
        For example:
          0.95 -> z ≈ 1.96
          0.68 -> z ≈ 1.0
          0.997 -> z ≈ 3.0

    Returns
    -------
    Rhythm
        Rhythm.x are the peaks aligned to the best φ,
        Rhythm.y are their amplitudes,
        Rhythm.period = mu
    """

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    assert x.shape == y.shape, "x and y must have same length"
    assert mu > 0.0, "mu must be > 0"
    assert sigma >= 0.0, "sigma must be >= 0"
    assert 0 < confidence < 1, "confidence must be in (0,1)"
    assert run_length > 0.0, "run_length must be > 0"

    # sort by time to keep behavior consistent
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # compute spread from sigma and confidence
    # For a symmetric interval [-spread, +spread], P(|Z| <= z) = confidence
    # => z = norm.ppf((1+confidence)/2)
    z = norm.ppf((1.0 + confidence) / 2.0)
    spread = z * sigma

    # candidate phases: x mod mu
    phases = np.mod(x_sorted, mu)

    best_phase = 0.0
    best_idx = np.array([], dtype=int)

    for phi in phases:
        # ideal ticks for this phase
        grid = np.arange(phi, run_length + mu, mu)

        matches = []
        gi = 0  # index over grid ticks
        oi = 0  # index over observed peaks

        while gi < len(grid) and oi < len(x_sorted):
            g_t = grid[gi]
            o_t = x_sorted[oi]

            if o_t < g_t - spread:
                oi += 1
            elif o_t > g_t + spread:
                gi += 1
            else:
                # |o_t - g_t| <= spread, assign this observed peak
                matches.append(oi)
                gi += 1
                oi += 1

        matches = np.asarray(matches, dtype=int)

        # keep the phase that explains the most peaks
        if matches.size > best_idx.size:
            best_phase = phi
            best_idx = matches
        elif matches.size == best_idx.size and matches.size > 0:
            # optional tie-breaker: pick the phase with lower avg abs error
            def avg_err(phi_candidate, idxs):
                g_local = np.arange(phi_candidate, run_length + mu, mu)
                errs = []
                for jj in idxs:
                    obs_t = x_sorted[jj]
                    nearest_tick = g_local[np.argmin(np.abs(g_local - obs_t))]
                    errs.append(abs(obs_t - nearest_tick))
                return np.mean(errs) if errs else np.inf

            cur_err = avg_err(phi, matches)
            best_err = avg_err(best_phase, best_idx)
            if cur_err < best_err:
                best_phase = phi
                best_idx = matches

    # build matched x,y for return
    if best_idx.size == 0:
        matched_x = np.array([], dtype=float)
        matched_y = np.array([], dtype=float)
    else:
        matched_x = x_sorted[best_idx]
        matched_y = y_sorted[best_idx]

    # ensure chronological order in final Rhythm
    if matched_x.size > 0:
        ord2 = np.argsort(matched_x)
        matched_x = matched_x[ord2]
        matched_y = matched_y[ord2]

    return Rhythm(
        x=matched_x,
        y=matched_y,
        period=mu,
        run_length=run_length,
        prefix="R"
    )

def gmpda_to_rhythms_from_periods(
    mu_list: Iterable[Union[int, float]],
    sigma_list: Iterable[Union[int, float]],
    x: np.ndarray,
    y: np.ndarray,
    sr: float,
    run_length: float,
    confidence: float = 0.95,
    prefix: str = "G",
) -> List[Rhythm]:
    """
    Convert GMPDA's (mu_list, sigma_list) into Rhythm objects you can score.

    For each (mu, sigma) pair from GMPDA:
    - convert mu (samples) -> mu_units (original x-units)
    - convert sigma (samples) -> sigma_units (original x-units)
    - call rhythm_from_mu with those values and the given confidence
    - tag the Rhythm with a name like "G0", "G1", ...

    Parameters
    ----------
    mu_list : iterable
        Periods from GMPDA in *samples*.
    sigma_list : iterable
        Sigmas from GMPDA in *samples*, same length/order as mu_list.
    x, y : np.ndarray
        Original peak times and heights in your domain units.
        These are what you fed into your detector / truth generator.
    sr : float
        Sample rate used to create ts (samples per unit).
        mu_units = mu_samples / sr, sigma_units = sigma_samples / sr.
    run_length : float
        Total time span of the signal (same units as x).
    confidence : float
        Confidence level to convert sigma -> spread (default 0.95).
    prefix : str
        Prefix for Rhythm.name, e.g. "G" -> G0, G1, ...

    Returns
    -------
    list[Rhythm]
        Rhythms aligned to each detected period, suitable for pairwise_matrix().
    """
    mu_list = list(mu_list)
    sigma_list = list(sigma_list)

    rhythms_out: List[Rhythm] = []

    for idx, (mu_samp, sig_samp) in enumerate(zip(mu_list, sigma_list)):
        # convert from samples -> your x units
        mu_units = float(mu_samp) / float(sr)
        sigma_units = float(sig_samp) / float(sr)

        r = rhythm_from_mu(
            x=x,
            y=y,
            mu=mu_units,
            sigma=sigma_units,
            run_length=run_length,
            confidence=confidence,
        )
        rhythms_out.append(r)

    return rhythms_out