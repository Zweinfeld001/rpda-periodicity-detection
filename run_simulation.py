from rhythm import make_rhythm, compose_rhythms
from rpda import get_detected_rhythms
from util import pairwise_matrix, matching_score
from util import plot_pulse_signal, plot_rhythm_peak_sets


T = 50

# --- Generate synthetic rhythms ---
r1 = make_rhythm(2.0, 1.0, T)
r2 = make_rhythm(3.2, 1.0, T)

true_rhythms = [r1, r2]

print("\n--- True Rhythms ---")
for i, r in enumerate(true_rhythms, start=1):
    print(f"R{i}: period={r.period}, peaks={len(r)}")

# --- Combine peaks into one signal ---
x, y = compose_rhythms(true_rhythms)

print("\n--- Signal Summary ---")
print("Total peaks:", len(x))
print("First 10 peak times:", x[:10])

# --- Plot the raw pulse signal ---
plot_pulse_signal(
    x,
    y,
    run_length=T,
    radius=0.02,
    title="Synthetic Pulse Signal"
)

# --- Run RPDA ---
detected = get_detected_rhythms(x, y, T)

print("\n--- Detected Rhythms ---")
print("Number detected:", len(detected))

for i, r in enumerate(detected, start=1):
    print(f"D{i}: period={r.period:.4f}, peaks={len(r)}")

# --- Plot detected rhythm assignments ---
plot_rhythm_peak_sets(
    x,
    y,
    detected,
    run_length=T,
    radius=0.02,
    title="Detected Rhythms"
)

# --- Pairwise similarity matrix ---
M = pairwise_matrix(true_rhythms, detected)

print("\n--- Pairwise Similarity Matrix (Jaccard) ---")
print(M)

# --- Matching score ---
score = matching_score(M.values)

print("\nJMS:", score)