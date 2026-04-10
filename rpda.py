import math
from typing import Dict, Any

import networkx as nx
from scipy.stats import norm, binom
from util import *


def get_x_sd() -> float:
    return 0.1

def get_y_sd() -> float:
    return 0.1

def calculate_x_margin(x_sd: float, confidence: float) -> float:
    z = norm.ppf(confidence)  # one-sided
    return z * x_sd

def calculate_y_margin(y_sd: float, confidence: float) -> float:
    z = norm.ppf(confidence)  # one-sided
    return z * y_sd

def calculate_p_null(x_margin: float, num_peaks: int, run_length: float) -> float:
    window_fraction = (2 * x_margin) / run_length
    p_null = 1 - (1 - window_fraction) ** num_peaks
    # print("window_fraction: ", window_fraction, "p_null: ", p_null)
    return min(p_null, 1.0)


class Candidate:
    def __init__(self, id_: int, d: float, anchor: float):
        self.id = id_
        self.d = d
        self.anchor = anchor
        self.hit_indices: Optional[List[int]] = None
        self.hits: Optional[int] = None
        self.tries: Optional[int] = None
        self.absorbed: List[int] = []

    def __repr__(self) -> str:
        res = f"ID: {self.id}, d: {self.d:.6f}, Anchor: {self.anchor:.6f}"
        if self.hits is not None and self.tries is not None:
            res += f", Hits: {self.hits}, Tries: {self.tries}, Indices: {self.hit_indices}"
        if self.absorbed:
            res += f", Absorbed: {self.absorbed}"
        return res

    def count_hits(self, x: np.ndarray, y: np.ndarray, x_margin: float, y_margin: float) -> None:
        x = np.asarray(x); y = np.asarray(y)

        # Anchor is a guaranteed hit (exact match assumed in your pipeline)
        anchor_idx = int(np.where(x == self.anchor)[0][0])

        hit_indices = [anchor_idx]
        tries = 1
        last_hit = self.anchor

        while True:
            t = last_hit + self.d
            if t > x[-1] + x_margin:
                break

            x_mask = (x >= t - x_margin) & (x <= t + x_margin)
            cand_idxs = np.where(x_mask)[0]

            if cand_idxs.size > 0:
                # Keep candidates whose y are within margin of recent mean
                recent_hits_y = y[hit_indices[-3:]]  # last up-to-3 hits
                expected_y = float(np.mean(recent_hits_y))
                valid = np.abs(y[cand_idxs] - expected_y) <= y_margin
                valid_idxs = cand_idxs[valid]

                if valid_idxs.size > 0:
                    # choose closest in x
                    distances = np.abs(x[valid_idxs] - t)
                    hit_idx = int(valid_idxs[np.argmin(distances)])
                    hit_indices.append(hit_idx)
                    last_hit = float(x[hit_idx])  # snap to actual
                else:
                    last_hit = float(t)
            else:
                last_hit = float(t)

            tries += 1

        self.hit_indices = hit_indices
        self.hits = len(hit_indices)
        self.tries = tries

    def binomial_test(self, p_null: float, alpha: float = 0.01) -> bool:
        if self.hits is None or self.tries is None:
            raise RuntimeError("Must call count_hits() before binomial_test().")
        if self.hits < 3:
            return False
        # skip first 2 guaranteed hits
        p_value = 1 - binom.cdf(self.hits - 2, self.tries - 2, p_null)
        return p_value < alpha


class Candidates:
    """
    Orchestrates the detection pipeline over global x/y (sorted),
    then exposes a simple conversion to Rhythm objects.
    """
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        run_length: float,
        confidence: float = 0.99,
        alpha: float = 0.01,
        x_sd_default: Optional[float] = None,
        y_sd_default: Optional[float] = None,
        verbose: bool = False,
    ):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.run_length = float(run_length)
        self.confidence = float(confidence)
        self.alpha = float(alpha)
        self.verbose = verbose

        # default SDs (for margins)
        self.x_sd_default = float(x_sd_default) if x_sd_default is not None else get_x_sd()
        self.y_sd_default = float(y_sd_default) if y_sd_default is not None else get_y_sd()

        # margins and null
        self.x_margin = calculate_x_margin(self.x_sd_default, self.confidence)
        self.y_margin = calculate_y_margin(self.y_sd_default, self.confidence)
        self.num_peaks = len(self.x)
        self.p_null = calculate_p_null(self.x_margin, self.num_peaks, self.run_length)

        # storage
        self.candidate_lst: List[Candidate] = []
        self.cur_id = 1

        # for verbosity/inspection
        self.last_similarity_df: Optional[pd.DataFrame] = None

    # Public driver (same 5 steps)
    def detect_regimes(
        self,
        similarity_threshold: float = 0.75,
        plot_radius: float = 0.01,
        show_plots: Optional[bool] = None,
    ) -> List[Rhythm]:
        """Run the full pipeline. If verbose, prints and visualizes along the way."""
        show_plots = self.verbose if show_plots is None else bool(show_plots)

        if self.verbose: print("Step 1: Generating candidates...")
        self._generate_candidates()

        if self.verbose: print("\nStep 2: Adding hit data...")
        self._add_hit_data()

        if self.verbose: print("\nStep 3: Pruning by binomial test...")
        self._prune_insufficient_hits()

        if self.verbose: print("\nStep 4: Grouping by hit similarity (overlap coefficient)...")
        self._group_candidates_by_similarity(threshold=similarity_threshold, show_plots=show_plots)

        if self.verbose: print("\nStep 5: Remove final outlier candidates with insufficient hits")
        self._remove_outliers()

        rhythms = self.to_rhythms()
        # Final overlay plot of the retained candidates
        if show_plots and len(self.candidate_lst) > 0:
            try:
                plot_rhythm_peak_sets(self.x, self.y, rhythms, self.run_length, radius=plot_radius)
            except NameError:
                # If the helper isn't in scope, don't fail—just note it.
                if self.verbose:
                    print("plot_rhythm_peak_sets(...) not found; skipping overlay plot.")

        if self.verbose:
            print("\nFinal number of remaining candidates:", self.get_num_regimes())

        return rhythms

    # ---- internal steps ----
    def _generate_candidates(self) -> None:
        for i in range(self._get_max_start_index()):
            for j in range(i + 1, self.num_peaks):
                d = round(self.x[j] - self.x[i], 4)
                if self.x[j] + self._get_min_hits() * d / 2 > self.run_length + self.x_margin:
                    break
                if d < 2 * self.x_margin or self.y[i] - self.y[j] > 2 * self.y_margin:
                    continue
                anchor = float(self.x[i])
                cand = Candidate(self.cur_id, d, anchor)
                self.candidate_lst.append(cand)
                self.cur_id += 1
                if self.verbose:
                    print(f"Generated candidate {cand.id}: d={d}, anchor={anchor}")
        if self.verbose:
            print(f"Total candidates generated: {len(self.candidate_lst)}")

    def _add_hit_data(self) -> None:
        for c in self.candidate_lst:
            c.count_hits(self.x, self.y, self.x_margin, self.y_margin)
            if self.verbose:
                print(f"Candidate {c.id}: d={c.d}, hits={c.hits}, tries={c.tries}, indices={c.hit_indices}")

    def _prune_insufficient_hits(self) -> None:
        original = len(self.candidate_lst)
        bonf = self.alpha / max(1, len(self.candidate_lst))  # Bonferroni
        self.candidate_lst = [
            c for c in self.candidate_lst
            if (c.hits or 0) >= 3 and c.binomial_test(self.p_null, alpha=bonf)
        ]
        if self.verbose:
            print(f"Pruned {original - len(self.candidate_lst)} candidates by significance")
            for c in self.candidate_lst:
                print(f"Candidate {c.id}: d={c.d}, hits={c.hits}, tries={c.tries}, indices={c.hit_indices}")

    def _group_candidates_by_similarity(self, threshold: float = 0.75, show_plots: bool = False) -> None:
        cands = self.candidate_lst
        n = len(cands)
        ids = [c.id for c in cands]

        # overlap coefficient on hit index sets
        def overlap(a: List[int], b: List[int]) -> float:
            if not a or b is None or not b:
                return 0.0
            sa, sb = set(a), set(b)
            inter = len(sa & sb)
            return inter / float(min(len(sa), len(sb)))

        # --- similarity matrix (for printing/inspection) ---
        if n > 0:
            M = np.zeros((n, n), dtype=float)
            for i in range(n):
                for j in range(n):
                    M[i, j] = overlap(cands[i].hit_indices, cands[j].hit_indices) if i != j else 1.0
            df = pd.DataFrame(M, index=ids, columns=ids)
            self.last_similarity_df = df
            if self.verbose:
                print("\nCandidate hit-index overlap matrix (Szymkiewicz–Simpson):")
                print(df.round(2))

        # --- graph (edges above threshold) ---
        G = nx.Graph()
        G.add_nodes_from(ids)
        for i in range(n):
            for j in range(i + 1, n):
                s = overlap(cands[i].hit_indices, cands[j].hit_indices)
                if s >= threshold:
                    G.add_edge(ids[i], ids[j])

        # visualize the graph when requested
        if show_plots and n > 1 and len(G) > 0:
            # Layout each connected component separately, then offset them
            components = list(nx.connected_components(G))
            pos: Dict[Any, np.ndarray] = {}
            component_spacing = 2.8  # distance between components on x-axis

            for idx, comp in enumerate(components):
                H = G.subgraph(comp)

                # Spring layout: central nodes tend toward center
                local_pos = nx.spring_layout(
                    H,
                    k=1.0 / np.sqrt(max(len(H), 1)),
                    iterations=150,
                    seed=0,
                )

                # Convert to array of coords
                coords = np.array([p for p in local_pos.values()])

                # Center at origin
                center = coords.mean(axis=0)
                coords -= center

                # De-elongate: make x and y have similar spread
                std = coords.std(axis=0)
                std[std == 0] = 1.0  # avoid divide-by-zero
                target_std = std.max()
                coords *= (target_std / std)

                # Normalize to a common radius (roughly unit circle)
                radii = np.sqrt((coords ** 2).sum(axis=1))
                max_r = radii.max() if radii.size > 0 else 1.0
                if max_r == 0:
                    max_r = 1.0
                coords /= max_r

                # Offset this component horizontally
                offset = np.array([idx * component_spacing, 0.0])
                for node, p in zip(local_pos.keys(), coords):
                    pos[node] = p + offset

            fig, ax = plt.subplots(figsize=(9, 6))

            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                node_size=450,
                node_color="lightblue",
                edgecolors="white",
                linewidths=0.8,
            )
            nx.draw_networkx_edges(
                G, pos, ax=ax,
                width=0.8,
                alpha=0.5,
            )
            nx.draw_networkx_labels(
                G, pos, ax=ax,
                font_size=8,
            )

            ax.set_title(f"Candidate Similarity Graph (threshold = {threshold:.2f})")
            ax.set_axis_off()
            ax.set_aspect("equal", "datalim")  # keep components from being stretched
            plt.tight_layout()
            plt.show()

        # --- keep strongest per connected component, absorb extras ---
        id_to_c = {c.id: c for c in cands}
        dominant: List[Candidate] = []
        centrality = nx.degree_centrality(G) if len(G) else {}

        for group in nx.connected_components(G) if len(G) else []:
            group_cands = [id_to_c[i] for i in group]
            dom = max(group_cands, key=lambda c: (centrality.get(c.id, 0.0), c.hits or 0))
            dom_hits = set(dom.hit_indices or [])
            absorbed = set()
            for other in group_cands:
                if other is dom:
                    continue
                absorbed.update(set(other.hit_indices or []) - dom_hits)
            dom.absorbed = sorted(absorbed)
            dominant.append(dom)
            if self.verbose and len(group_cands) > 1:
                print(
                    f"Merged group {[c.id for c in group_cands]} into candidate {dom.id}. "
                    f"Absorbed {len(dom.absorbed)} extra hit indices."
                )

        grouped_ids = set(c.id for c in dominant)
        singles = [
            c for c in cands
            if (len(G) == 0 or G.degree[c.id] == 0) and c.id not in grouped_ids
        ]
        self.candidate_lst = dominant + singles
        if self.verbose:
            print(f"Retaining {len(self.candidate_lst)} dominant/single candidate(s).")

    def _remove_outliers(self) -> None:
        min_hits = self._get_min_hits()
        original = len(self.candidate_lst)
        self.candidate_lst = [
            c for c in self.candidate_lst
            if len(c.hit_indices or []) + len(c.absorbed or []) >= min_hits
        ]
        if self.verbose:
            print(f"Removed {original - len(self.candidate_lst)} outlier candidate(s) (min_hits = {min_hits})")

    # ---- helpers ----
    def _get_max_start_index(self) -> int:
        return min(int(self.num_peaks / 2), self.num_peaks - 1)

    def _get_min_hits(self) -> int:
        return max(math.ceil(math.sqrt(self.num_peaks)), 1)

    def get_num_regimes(self) -> int:
        return len(self.candidate_lst)

    # ---- export final candidates as rhythms ----
    def to_rhythms(
        self,
        include_absorbed: bool = True,
        name_prefix: str = "C",
    ) -> List[Rhythm]:
        rhythms: List[Rhythm] = []
        for j, c in enumerate(self.candidate_lst):
            idxs = list(getattr(c, "hit_indices", []) or [])
            if include_absorbed:
                idxs += list(getattr(c, "absorbed", []) or [])
            if not idxs:
                continue
            idxs = np.asarray(sorted(set(idxs)), dtype=int)  # unique & sorted
            x_ev = self.x[idxs]
            y_ev = self.y[idxs]
            period = float(getattr(c, "d", np.nan))
            rhythms.append(Rhythm(x_ev, y_ev, period=period,
                                  run_length=self.run_length, prefix="C"))
        return rhythms



def get_detected_rhythms(x: np.ndarray, y: np.ndarray, run_length: float, verbose: bool = False
                         ) -> List[Rhythm]:
    if len(x) < 3:  # require at least 3 peaks
        return []
    det = Candidates(x, y, run_length, verbose=verbose)
    rhythms = det.detect_regimes()
    return rhythms