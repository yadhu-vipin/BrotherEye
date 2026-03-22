# EXAMPLE.md — Manual Walkthrough of State Transitions

---

## Setup

**Building b1**

| Zone Index | Zone Label | Role |
|---|---|---|
| 0 | z1_b1 | Entry — camera captures arrivals |
| 1 | z2_b1 | Internal |
| 2 | z3_b1 | Internal |
| 3 | z4_b1 | Exit — camera captures departures |
| 4 | zT_b1 | Outdoor transition area between buildings |

Registered occupants: `person_a`, `person_b`

**Building b2**

| Zone Index | Zone Label | Role |
|---|---|---|
| 0 | z1_b2 | Entry — camera captures arrivals |
| 1 | z2_b2 | Internal |
| 2 | z3_b2 | Internal |
| 3 | z4_b2 | Exit — camera captures departures |
| 4 | zT_b2 | Outdoor transition area between buildings |

Registered occupants: `person_c`, `person_d`

**Movement path through a building:**
```
zT  →  z1 (entry)  →  z2  →  z3  →  z4 (exit)  →  zT
```

---

## Initial State s0 (before any event)

Every occupant starts at probability 1.0 at their entry zone z1. No camera has fired yet.

**b1 — Registered State Table — s0**

| | z1_b1 | z2_b1 | z3_b1 | z4_b1 | zT_b1 |
|---|---|---|---|---|---|
| person_a | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| person_b | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**b2 — Registered State Table — s0**

| | z1_b2 | z2_b2 | z3_b2 | z4_b2 | zT_b2 |
|---|---|---|---|---|---|
| person_c | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| person_d | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

---

## Event e1 — person_a detected at z1_b1 (entry)

```
image      :  dataset/person_a/test/img_01.jpg
building   :  b1
zone       :  z1_b1  (index 0)
timestamp  :  09:05:00
```

**Recognition pipeline:**

| Step | Module | Result |
|---|---|---|
| Face detection | `batch_face_locations` | Face bounding box found |
| Embedding | `face_encodings` | 128-d vector extracted |
| Distance scoring | `face_distance` | person_a → 0.08,  person_b → 0.61 |
| Validation | `compare_faces` | person_a → 18 True,  person_b → 0 True |
| Match | — | person_a confirmed |

**Probability mapping:**

$$p = e^{-5.0 \times 0.08} = e^{-0.40} = 0.670$$

**Event tuple e1** — $e^u_k = \langle t,\ z^u_j,\ P_k \rangle$

| Field | Value |
|---|---|
| t | 09:05:00 |
| z^u_j | z1_b1 |
| P_k | { person_a: 0.670,  person_b: 0.005 } |

---

## State s0 → s1

Detection at j = 0 (z1_b1), p = 0.670, (1 − p) = 0.330

**Transition applied to person_a:**

| Zone | Formula | Calculation | New Value |
|---|---|---|---|
| z1_b1 (j=0) | Eq. 9:  p + (1−p) × old | 0.670 + 0.330 × 1.000 | 1.000 |
| z2_b1 | Eq. 10: (1−p) × old | 0.330 × 0.000 | 0.000 |
| z3_b1 | Eq. 10: (1−p) × old | 0.330 × 0.000 | 0.000 |
| z4_b1 | Eq. 10: (1−p) × old | 0.330 × 0.000 | 0.000 |
| zT_b1 | Eq. 10: (1−p) × old | 0.330 × 0.000 | 0.000 |

Sum: 1.000 ✓

**b1 — Registered State Table — s1**

| | z1_b1 | z2_b1 | z3_b1 | z4_b1 | zT_b1 |
|---|---|---|---|---|---|
| person_a | **1.000** | 0.000 | 0.000 | 0.000 | 0.000 |
| person_b | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

No visible change — person_a was already assumed to be at z1. The event confirms their position.

---

## Event e2 — person_a detected at z3_b1 (internal)

```
image      :  dataset/person_a/test/img_02.jpg
building   :  b1
zone       :  z3_b1  (index 2)
timestamp  :  09:31:00
```

| Step | Module | Result |
|---|---|---|
| Distance scoring | `face_distance` | person_a → 0.05,  person_b → 0.58 |
| Validation | `compare_faces` | person_a → 19 True,  person_b → 0 True |
| Match | — | person_a confirmed |

$$p = e^{-5.0 \times 0.05} = e^{-0.25} = 0.779$$

**Event tuple e2:**

| Field | Value |
|---|---|
| t | 09:31:00 |
| z^u_j | z3_b1 |
| P_k | { person_a: 0.779,  person_b: 0.004 } |

---

## State s1 → s2

Detection at j = 2 (z3_b1), p = 0.779, (1 − p) = 0.221

**Transition applied to person_a:**

| Zone | Formula | Calculation | New Value |
|---|---|---|---|
| z1_b1 | Eq. 10: (1−p) × old | 0.221 × 1.000 | 0.221 |
| z2_b1 | Eq. 10: (1−p) × old | 0.221 × 0.000 | 0.000 |
| z3_b1 (j=2) | Eq. 9:  p + (1−p) × old | 0.779 + 0.221 × 0.000 | 0.779 |
| z4_b1 | Eq. 10: (1−p) × old | 0.221 × 0.000 | 0.000 |
| zT_b1 | Eq. 10: (1−p) × old | 0.221 × 0.000 | 0.000 |

Sum: 0.221 + 0.779 = 1.000 ✓

**b1 — Registered State Table — s2**

| | z1_b1 | z2_b1 | z3_b1 | z4_b1 | zT_b1 |
|---|---|---|---|---|---|
| person_a | **0.221** | 0.000 | **0.779** | 0.000 | 0.000 |
| person_b | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

person_a is 77.9% likely at z3. The 22.1% residual at z1 reflects that no camera confirmed they left z1 — only that they were seen at z3.

---

## Event e3 — person_a detected at z4_b1 (exit zone)

```
image      :  dataset/person_a/test/img_03.jpg
building   :  b1
zone       :  z4_b1  (index 3)
timestamp  :  10:10:00
```

| Step | Module | Result |
|---|---|---|
| Distance scoring | `face_distance` | person_a → 0.07,  person_b → 0.60 |
| Validation | `compare_faces` | person_a → 16 True,  person_b → 0 True |
| Match | — | person_a confirmed |

$$p = e^{-5.0 \times 0.07} = e^{-0.35} = 0.705$$

**Event tuple e3:**

| Field | Value |
|---|---|
| t | 10:10:00 |
| z^u_j | z4_b1 |
| P_k | { person_a: 0.705,  person_b: 0.005 } |

---

## State s2 → s3

Detection at j = 3 (z4_b1), p = 0.705, (1 − p) = 0.295

**Transition applied to person_a:**

| Zone | Formula | Calculation | New Value |
|---|---|---|---|
| z1_b1 | Eq. 10: (1−p) × old | 0.295 × 0.221 | 0.065 |
| z2_b1 | Eq. 10: (1−p) × old | 0.295 × 0.000 | 0.000 |
| z3_b1 | Eq. 10: (1−p) × old | 0.295 × 0.779 | 0.230 |
| z4_b1 (j=3) | Eq. 9:  p + (1−p) × old | 0.705 + 0.295 × 0.000 | 0.705 |
| zT_b1 | Eq. 10: (1−p) × old | 0.295 × 0.000 | 0.000 |

Sum: 0.065 + 0.000 + 0.230 + 0.705 + 0.000 = 1.000 ✓

**b1 — Registered State Table — s3**

| | z1_b1 | z2_b1 | z3_b1 | z4_b1 | zT_b1 |
|---|---|---|---|---|---|
| person_a | **0.065** | 0.000 | **0.230** | **0.705** | 0.000 |
| person_b | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

person_a is 70.5% likely at z4 (exit door). The residuals at z1 and z3 have been shrinking with each new event as Eq. 10 keeps scaling them down. zT is still 0 because person_a has not been detected outside yet.

---

## Event e4 — person_a detected at zT_b1 (outdoor transition area)

```
image      :  dataset/person_a/test/img_04.jpg
building   :  b1
zone       :  zT_b1  (index 4)
timestamp  :  10:12:00
```

| Step | Module | Result |
|---|---|---|
| Distance scoring | `face_distance` | person_a → 0.06,  person_b → 0.63 |
| Validation | `compare_faces` | person_a → 17 True,  person_b → 0 True |
| Match | — | person_a confirmed |


**Event tuple e4:**

| Field | Value |
|---|---|
| t | 10:12:00 |
| z^u_j | zT_b1 |
| P_k | { person_a: 0.741,  person_b: 0.005 } |

---

## State s3 → s4

Detection at j = 4 (zT_b1), p = 0.741, (1 − p) = 0.259

**Transition applied to person_a:**

| Zone | Formula | Calculation | New Value |
|---|---|---|---|
| z1_b1 | Eq. 10: (1−p) × old | 0.259 × 0.065 | 0.017 |
| z2_b1 | Eq. 10: (1−p) × old | 0.259 × 0.000 | 0.000 |
| z3_b1 | Eq. 10: (1−p) × old | 0.259 × 0.230 | 0.060 |
| z4_b1 | Eq. 10: (1−p) × old | 0.259 × 0.705 | 0.183 |
| zT_b1 (j=4) | Eq. 9:  p + (1−p) × old | 0.741 + 0.259 × 0.000 | 0.741 |

Sum: 0.017 + 0.000 + 0.060 + 0.183 + 0.741 = 1.001 → renormalised → **1.000 ✓**

**b1 — Registered State Table — s4**

| | z1_b1 | z2_b1 | z3_b1 | z4_b1 | zT_b1 |
|---|---|---|---|---|---|
| person_a | **0.017** | 0.000 | **0.060** | **0.183** | **0.741** |
| person_b | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

person_a is 74.1% likely to be in the outdoor transition zone. The residuals at z1, z3, z4 are the accumulated uncertainty from earlier detections, all being progressively scaled down by Eq. 10 with each new event.

---

## Event e5 — person_a detected at z1_b2 (visitor arrival at b2)

```
image      :  dataset/person_a/test/img_05.jpg
building   :  b2
zone       :  z1_b2  (index 0)
timestamp  :  10:14:00
```

b2's local recogniser checks against `person_c` and `person_d` — no match. Escalates to master registry — matches `person_a`.

**person_a is flagged as a visitor to b2.**

| Action | Detail |
|---|---|
| Local match in b2 | Failed — person_a not in b2's registered list |
| Master registry lookup | person_a matched, home building = b1 |
| Visitor row added | New row for person_a in b2's visitor table, initialised at z1_b2 = 1.0 |
| b1 home update | person_a's zT_b1 set to 0.980, all internal zones scaled down — confirms departure |

**b1 — Registered State Table after visitor confirmation**

| | z1_b1 | z2_b1 | z3_b1 | z4_b1 | zT_b1 |
|---|---|---|---|---|---|
| person_a | 0.005 | 0.000 | 0.005 | 0.010 | **0.980** |
| person_b | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**b2 — Visitor State Table (new row added)**

| | z1_b2 | z2_b2 | z3_b2 | z4_b2 | zT_b2 |
|---|---|---|---|---|---|
| person_a | **1.000** | 0.000 | 0.000 | 0.000 | 0.000 |

---

## Global State after e5 (assembled on demand)

| Occupant | Table | z1 | z2 | z3 | z4 | zT |
|---|---|---|---|---|---|---|
| person_a | b1 registered | 0.005 | 0.000 | 0.005 | 0.010 | 0.980 |
| person_b | b1 registered | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| person_a | b2 visitor | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| person_c | b2 registered | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| person_d | b2 registered | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |

As a flat r-tuple:

```
( 0.005, 0.000, 0.005, 0.010, 0.980,    ← person_a in b1 registered
  1.000, 0.000, 0.000, 0.000, 0.000,    ← person_b in b1 registered
  1.000, 0.000, 0.000, 0.000, 0.000,    ← person_a in b2 visitor
  1.000, 0.000, 0.000, 0.000, 0.000,    ← person_c in b2 registered
  1.000, 0.000, 0.000, 0.000, 0.000 )   ← person_d in b2 registered
```

This tuple is temporary — assembled only because a query was issued. It is not stored.

---

## Full State Transition Summary

| State | Event | Zone | p | What changed |
|---|---|---|---|---|
| s0 | — | — | — | All occupants at 1.0 in z1 of home building |
| s1 | e1 | z1_b1 (entry) | 0.670 | person_a confirmed at z1 — no visible change |
| s2 | e2 | z3_b1 (internal) | 0.779 | person_a: z1→0.221, z3→0.779 |
| s3 | e3 | z4_b1 (exit) | 0.705 | person_a: z1→0.065, z3→0.230, z4→0.705 |
| s4 | e4 | zT_b1 (outdoor) | 0.741 | person_a: z1→0.017, z3→0.060, z4→0.183, zT→0.741 |
| s5 | e5 | z1_b2 (visitor) | — | person_a: zT_b1→0.980 + new visitor row in b2 at z1→1.000 |

---

## Equations Used

**Eq. 9 — detected zone j:**

$$Z^u_{jk} = p + (1 - p) \times Z^u_{j,k-1}$$

**Eq. 10 — all other zones l ≠ j:**

$$Z^u_{lk} = (1 - p) \times Z^u_{l,k-1}$$

**Eq. 6 — probability constraint verified after every update:**

$$\sum_{j=1}^{m_u} p^u_{jk}(o^u_i)\ +\ p^u_{Tk}(o^u_i)\ =\ 1$$

**Distance to probability mapping:**

$$p = e^{-5.0 \times d}$$

where d is the euclidean distance returned by `face_distance`.