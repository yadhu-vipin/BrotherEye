# DSTS Simulation — Mohan & Menon, IET Computer Vision 2020

> **"Modelling large scale camera networks for identification and tracking: an abstract framework"**
> Lakshmi Mohan, Vivek Menon — *IET Computer Vision, Vol. 14, Iss. 7, pp. 426–433, 2020*
> DOI: [10.1049/iet-cvi.2019.0959](https://doi.org/10.1049/iet-cvi.2019.0959)

A full Python implementation of the **Distributed State Transition System (DSTS)** model proposed in the paper, including all mathematical definitions, the face recognition pipeline, visitor roaming logic, temporal event ordering, and precision–recall evaluation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Implementation Details](#implementation-details)
- [Language and Tools](#language-and-tools)
- [File Structure](#file-structure)
- [How to Run](#how-to-run)
- [Simulation Output Walkthrough](#simulation-output-walkthrough)
- [Paper → Code Mapping](#paper--code-mapping)
- [Mock vs Real Mode](#mock-vs-real-mode)
- [Known Limitations](#known-limitations)

---

## Overview

The paper proposes a distributed approach to tracking occupants across a **wide-area, multi-building indoor surveillance environment** (e.g., a university campus). Instead of maintaining one giant centralised state table for all buildings, the system decomposes the problem into independent **Building-Specific State Transition Systems (BSTS)** that coordinate with each other under a global **Distributed State Transition System (DSTS)** coordinator.

This implementation faithfully reproduces the paper's experimental setup:

| Parameter | Value |
|---|---|
| Number of buildings | 2 |
| Building b1 zones | z1, z2, z3, z4, zT (transition) |
| Building b2 zones | z1, z2, z3, zT (transition) |
| Registered occupants in b1 | 15 |
| Registered occupants in b2 | 20 |
| Active occupants simulated | 10 (from b1) |
| Total recognition events | 86 (+ 2 visitor events) |
| Simulation time window | 09:00 AM – 04:30 PM |
| Face recognition backbone | Dlib ResNet-34 (99.38% on LFW) |
| Default recognition threshold θ | 0.82 |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         DSTS                                │
│               (Global Coordinator)                          │
│                                                             │
│   Global Event Log  ──  Ordered by timestamp + tie-break   │
│   Master Registry   ──  occupant_id → home_building_id      │
│   r-tuple State     ──  Eq. 4 & 5 (aggregated BSTS states)  │
│                                                             │
│      ┌──────────────┐          ┌──────────────┐            │
│      │    BSTS b1   │          │    BSTS b2   │            │
│      │              │          │              │            │
│      │ Registered   │          │ Registered   │            │
│      │ State Table  │          │ State Table  │            │
│      │ (15 × 5)     │          │ (20 × 4)     │            │
│      │              │          │              │            │
│      │ Visitor      │          │ Visitor      │            │
│      │ State Table  │◄─roam───►│ State Table  │            │
│      │ (dynamic)    │          │ (dynamic)    │            │
│      │              │          │              │            │
│      │  Δ^u(Eq9,10) │          │  Δ^u(Eq9,10) │            │
│      └──────────────┘          └──────────────┘            │
│                                                             │
│              FaceRecognitionEngine                          │
│         (Real: Dlib / Mock: Gaussian noise)                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### 1. State Tables (`StateTable`, `OccupantEntry`)

Each BSTS maintains two state tables as defined in the paper:

- **Registered Table** (`s_R^u`) — Fixed rows for occupants who belong to this building. Size: `n_u × (m_u + 1)`, where the `+1` column is the transition zone.
- **Visitor Table** (`s_V^u`) — Dynamic rows that appear when an occupant from another building is detected (roaming). Rows are added on the fly.

Each row stores a probability vector satisfying **Equation 6**:

```
∑ p_jk^u(o_i^u)  +  p_Tk^u(o_i^u)  =  1.0
  j=1..m_u
```

This constraint is verified after every state update to catch floating-point drift.

---

### 2. Transition Function (`BSTS.apply_transition`)

When a recognition event fires at zone `j` with probability `p` for occupant `o_i`, the state is updated using **Equations 9 and 10**:

```
# Eq. 9 — detected zone j:
Z^u_jk  =  p  +  (1 - p) × Z^u_{j,k-1}

# Eq. 10 — all other zones l ≠ j:
Z^u_lk  =  (1 - p) × Z^u_{l,k-1}
```

After applying these equations, probabilities are renormalised to guard against accumulated floating-point error. The constraint check (`verify_constraint()`) asserts the sum remains `1.0` to within `1e-9`.

---

### 3. Recognition Events (`RecognitionEvent`)

Each event corresponds to the paper's definition:

```
e^u_k  =  (t,  z^u_j,  P_k)
```

where `P_k = { (o_i, p_jk(o_i)) : 1 ≤ i ≤ n_u }` is a dictionary mapping every tracked occupant to their recognition probability at the event. Only the matched occupant receives a high probability; all others receive a small residual drawn from `Uniform(0.001, 0.01)`.

---

### 4. Face Recognition Engine (`FaceRecognitionEngine`)

**Real mode** (when `face_recognition` is installed):

1. `batch_face_locations` — CNN-based face detector returns bounding boxes.
2. `face_encodings` — ResNet-34 generates a 128-dimensional embedding.
3. `face_distance` — Euclidean distance against all registered encodings.
4. `compare_faces` — Boolean vote with built-in tolerance threshold.
5. **Genuine match criterion:** minimum **12 TRUE votes** (as reported in paper Section 4).
6. **Distance → Probability mapping:** `p = exp(-5.0 × d)`, an exponential decay giving `p ≈ 0.99` at `d = 0` and `p ≈ 0.05` at `d = 0.6`.

**Mock mode** (fallback when the library is absent):

Recognition probability is sampled from `N(0.92, 0.05)`, clipped to `[0.5, 0.999]`. The simulation logic, all equations, and all data structures are **identical** in both modes. Only the probability value changes source.

---

### 5. Visitor / Roaming Logic (`DSTS._handle_visitor_arrival`)

Inspired by the cellular network analogy in the paper:

- Every occupant has a **home building** (analogous to home network).
- When a visitor is detected at a foreign building, the face recogniser fails to match locally and escalates to the **Master Database** lookup across all 35 occupants.
- On confirmed visitor arrival:
  - A new row is added to the **Visitor State Table** of the host building.
  - The **home building's registered table** sets the occupant's transition zone probability to `0.98`, confirming they have left.

---

### 6. Temporal Ordering (`DSTS.happened_before`, `DSTS._insert_ordered`)

The paper defines two levels of ordering (Section 3.3):

**Partial order** — "happened before" relation (→), Equation 12:
- Within the same BSTS: `e_k → e_k'` if `k < k'` (sequential event counter).
- Across different BSTS: order is inferred from physical timestamps.
- Two events are **concurrent** (`∥`) if neither happened before the other.

**Total order** — tie-breaking when physical timestamps are equal:
1. Primary: timestamp (ascending)
2. Secondary: building ID alphabetical order (b1 before b2)
3. Tertiary: event counter within the BSTS

---

### 7. Global State (`DSTS.global_state`)

Implements Equations 4 and 5 — the DSTS state at time `t` is obtained by aggregating the most recent BSTS states at or before `t` into a single flat r-tuple:

```
S'_k,t  =  ⟨ s^u_{ku}, t' ⟩   where  t' ≤ t,  for all 1 ≤ u ≤ w
```

If no events have occurred in a BSTS during `[t', t]`, the last known state is carried forward unchanged.

---

### 8. Precision & Recall Evaluation (`DSTS.evaluate_state`, `DSTS.evaluate_average`)

Reproduces Tables 5 and 6 from the paper. For a given recognition threshold `θ`:

```
ak  = tp   (recognised AND in ground truth)
bk  = tp + fp  (all recognised, prob ≥ θ)
ck  = tp + fn  (all in ground truth)

Precision  =  ak / bk
Recall     =  ak / ck
```

Average precision and recall are computed across all 86 states by evaluating each one independently and averaging.

---

## Language and Tools

| Category | Tool / Library | Version | Purpose |
|---|---|---|---|
| Language | **Python** | 3.8+ | Core implementation |
| Face recognition | **face_recognition** | 1.3.0 | High-level Dlib wrapper |
| Deep learning backbone | **Dlib** | 19.x | ResNet-34 face embedding (128-d) |
| Image I/O | **Pillow (PIL)** | 9.x | Load and decode image files |
| Numerical | **NumPy** | 1.x | Face encodings and distance computation |
| Standard library | `dataclasses` | — | Clean data container definitions |
| Standard library | `typing` | — | Type hints throughout |
| Standard library | `math`, `random` | — | Probability mapping, mock noise |
| Standard library | `time`, `datetime` | — | Physical clock timestamps |
| Standard library | `copy` | — | State snapshotting for time queries |

> **No external frameworks** (PyTorch, TensorFlow, OpenCV) are required. The face_recognition library bundles its own pre-trained model weights.

---

## File Structure

```
.
├── dsts_simulation.py      ← Full implementation (single self-contained file)
└── README.md               ← This file
```

The entire simulation is intentionally kept in one file so the mapping from paper equations to code is easy to follow linearly.

---

## How to Run

### Minimal (Mock Mode — no dependencies)

```bash
python dsts_simulation.py
```

Runs immediately with no installs. Recognition probabilities are simulated stochastically.

### Full (Real Face Recognition Mode)

```bash
# Install dependencies
pip install face_recognition

# Prepare face images (40 images per occupant, as in the paper)
# Place them at:  ./faces/<occupant_id>/<image_n>.jpg

# Run
python dsts_simulation.py
```

The engine automatically switches to real mode when `face_recognition` is importable and the image path passed to `submit_recognition()` resolves to an actual file.

### Integrating into Your Own Code

```python
from dsts_simulation import DSTS, BSTS, FaceRecognitionEngine

# Build buildings
bsts1 = BSTS("b1", ["z1_b1", "z2_b1", "z3_b1", "zT_b1"], ["alice", "bob"])
bsts2 = BSTS("b2", ["z1_b2", "z2_b2", "zT_b2"], ["carol"])

# Wire up DSTS
dsts = DSTS()
dsts.register_building(bsts1)
dsts.register_building(bsts2)
dsts.set_recognition_engine(FaceRecognitionEngine({}))

# Submit a recognition event
event = dsts.submit_recognition(
    building_id="b1",
    zone_index=0,
    image_path_or_id="alice",   # real image path in production
    mock_true_id="alice"
)

# Query location
result = dsts.query_occupant_location("alice", theta=0.82)
# → ("b1", "z1_b1", 0.9273)
```

---

## Simulation Output Walkthrough

Running the script produces the following sections in order:

```
1. DSTS Global State
   ├── Building b1 Registered Table  (15 occupants × 5 zones)
   └── Building b2 Visitor Table     (visitor row for roaming occupant)

2. Occupant Location Queries  (θ = 0.82)
   └── Each queried occupant → building, zone, probability

3. Occupant Track  (Fig. 6 equivalent)
   └── Timestamped movement log for the roaming occupant across b1 → b2

4. Table 5 — State-level Precision & Recall
   └── tp, fp, fn, ak, bk, ck, Precision, Recall at each θ

5. Table 6 — Average Precision & Recall
   └── Averaged over all 86 states, for each θ

6. Event Ordering Demo
   └── Demonstrates the happened-before (→) and concurrency (∥) relations

7. Global Event Log (last 10 entries)
   └── Timestamp, building, zone, top-3 probability occupants
```

---

## Paper → Code Mapping

| Paper Element | Code Location |
|---|---|
| Eq. 1 — Centralised STS transition `Δ: S×E → S` | Background context only (not implemented; DSTS replaces it) |
| Eq. 2 — Zone state tuple `Z^u_jk` | `OccupantEntry.probs`, `StateTable` |
| Eq. 3 — BSTS state `s^u_ku = s_R ∪ s_V` | `BSTS.current_state()` returning `(registered_table, visitor_table)` |
| Eq. 4 — DSTS global state at time t | `DSTS.global_state(t)` |
| Eq. 5 — DSTS state when no events in interval | `DSTS.get_state_at_time()` carry-forward logic |
| Eq. 6 — Probability constraint | `OccupantEntry.verify_constraint()` |
| Eq. 7 — DSTS event set `E' = ∪ E^u` | `DSTS.global_event_log` |
| Eq. 8 — BSTS transition `Δ^u: S^u × E^u → S^u` | `BSTS.apply_transition()` |
| Eq. 9 — Update detected zone j | `apply_transition()` branch `l == j` |
| Eq. 10 — Update all other zones l ≠ j | `apply_transition()` branch `l != j` |
| Eq. 11 — DSTS transition `Δ': S'×E' → S'` | `DSTS.submit_recognition()` (delegates to BSTS) |
| Eq. 12 — Causal precedence within BSTS | `DSTS.happened_before()` |
| Table 1 — Registered state table | `BSTS.registered_table` (`StateTable`) |
| Table 2 — Visitor state table | `BSTS.visitor_table` (`StateTable`) |
| Table 5 — State-level precision & recall | `DSTS.evaluate_state()` |
| Table 6 — Average precision & recall | `DSTS.evaluate_average()` |
| Fig. 2 — DSTS space-time diagram | `DSTS.happened_before()` / `are_concurrent()` |
| Fig. 3 — Event ordering | `DSTS._insert_ordered()` (total order tie-breaking) |
| Fig. 6 — Occupant track across buildings | `DSTS.generate_occupant_track()` |
| Section 3.2 — Visitor / roaming analogy | `DSTS._handle_visitor_arrival()` |
| Section 4 — Face recognition pipeline | `FaceRecognitionEngine.recognize_from_image()` |

---

## Mock vs Real Mode

| Feature | Mock Mode | Real Mode |
|---|---|---|
| Requires `face_recognition` | No | Yes |
| Requires face image files | No | Yes (40 per occupant) |
| State transition logic | Identical | Identical |
| Probability source | `N(0.92, 0.05)` | `exp(-5d)` from Dlib distance |
| Visitor detection | Simulated via flag | Real face embedding lookup |
| All equations active | Yes | Yes |
| Suitable for | Code study, unit testing | Full deployment / real data |

---

## Known Limitations

- **Unknown individuals** are not tracked. The paper acknowledges this is a separate research problem (Section 4).
- **State snapshots** for historical queries (`get_state_at_time`) return the current state as an approximation. A production system would snapshot after each event.
- **b2 occupants** are initialised but not given movement events in the default simulation, matching the paper's focus on b1 active occupants.
- The **distance-to-probability mapping** (`exp(-5d)`) is one reasonable choice; the paper references the approach from the 3R's model [Ref 7] without specifying the exact function.