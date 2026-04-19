# RPDA: Rhythmic Periodicity Detection in Sparse Event Sequences

Implementation of the **Rhythmic Periodicity Detection Algorithm (RPDA)** for identifying and decomposing periodic structure in sparse impulsive event sequences.

This repository accompanies the paper:

**Zachary Weinfeld, Kelly Bodwin, Alex Dekhtyar, Michael Khasin**  
*Rhythmic Periodicity Detection in Sparse Impulsive Event Sequences*

---

## Overview

Many physical and engineered systems produce signals consisting of **sparse impulsive events** rather than continuous waveforms. Traditional spectral methods often struggle to recover periodic structure from such data.

RPDA is a peak-based algorithm that detects rhythmic structure directly from event sequences by:

1. Constructing candidate rhythms from inter-event spacings  
2. Counting forward matches along predicted rhythm grids  
3. Testing statistical significance using a binomial null model  
4. Grouping overlapping candidates via the overlap coefficient  
5. Returning rhythmic components with sufficient support  

The algorithm outputs:

- detected rhythms
- estimated periods
- peak assignments for each rhythm

---

## Repository Structure

rpda-periodicity-detection/

README.md  
requirements.txt  

rpda.py            # Core RPDA detection algorithm  
rhythm.py          # Rhythm object and synthetic data generation  
util.py            # Metrics, visualization, evaluation tools  
gmpda_local.py     # GMPDA baseline comparison  

run_simulation.py  # Example experiment  

---

## Installation

Clone the repository:
```
git clone https://github.com/zacharyweinfeld/rpda-periodicity-detection.git  
cd rpda-periodicity-detection
```

Install dependencies:
```
pip install -r requirements.txt
```

---

## Quick Example

```python
from rhythm import make_rhythm, compose_rhythms
from rpda import get_detected_rhythms

T = 50

# Generate synthetic rhythms
r1 = make_rhythm(2.0, 1.0, T)
r2 = make_rhythm(3.2, 1.0, T)

# Combine into one signal
x, y = compose_rhythms([r1, r2])

# Run RPDA
detected = get_detected_rhythms(x, y, T)

for r in detected:
    print("Detected period:", r.period)
```

---

## Visualization

Example:

```python
from util import plot_rhythm_peak_sets

plot_rhythm_peak_sets(x, y, detected, run_length=T)
```

---

## Evaluation Metrics

Example:

```python
from util import pairwise_matrix, matching_score

M = pairwise_matrix(true_rhythms, detected)
score = matching_score(M.values)

print("Jaccard Matching Score:", score)
```

---

## Reproducing Experiments

```
python run_simulation.py
```

This script:

1. Generates two synthetic rhythms
2. Runs RPDA
3. Computes evaluation metrics
4. Produces diagnostic plots

---

## Requirements

numpy  
scipy  
pandas  
matplotlib  
networkx  

Install via:

```
pip install -r requirements.txt
```

---

## Citation

Weinfeld, Z., Bodwin, K., Dekhtyar, A., & Khasin, M.  
Rhythmic Periodicity Detection in Sparse Impulsive Event Sequences.

---

## Contact

Zachary Weinfeld  
zacharyweinfeld@gmail.com
