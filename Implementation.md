# IMPLEMENTATION.md

---

## Libraries Used

| Library | Purpose |
|---|---|
| `face_recognition` | Face detection, 128-d embedding extraction, distance scoring, compare_faces voting |
| `numpy` | Stores and operates on 128-d embedding arrays |
| `Pillow (PIL)` | Loads image files before passing to face_recognition |
| `os` | Traversing the dataset folder structure |
| `dataclasses` | Defining `OccupantEntry` and `RecognitionEvent` as clean structs |
| `math` | `exp()` for mapping distance scores to probabilities |
| `time`, `datetime` | Attaching physical timestamps to every recognition event |
| `copy` | Deep-copying state tables when snapshotting historical states |
| `typing` | Type hints — `Dict`, `List`, `Optional`, `Tuple` |

---

## Dataset Layout

40 photos per person, manually collected, split into two halves:

```
dataset/
├── person_a/
│   ├── train/     ← 20 photos  —  used once to generate stored encodings
│   └── test/      ← 20 photos  —  fed one at a time as simulated CCTV frames
├── person_b/ ...
├── person_c/ ...
└── person_d/ ...
```

`person_a`, `person_b` → registered in **b1**
`person_c`, `person_d` → registered in **b2**

---

## The Face Recognition Model

Uses a **ResNet-34** network pre-trained on the Labelled Faces in the Wild dataset (99.38% accuracy). The model is **never retrained or fine-tuned**. It is a fixed feature extractor — it takes a face image and returns a 128-d vector encoding the facial geometry. The model weights are frozen throughout the entire simulation.

---

## Registration Phase (done once, before simulation)

Each photo in a person's `train/` folder is passed through the frozen ResNet-34. The model returns one 128-d numpy array per photo. All 20 vectors are stored as that person's encodings in the master registry.

This is not learning. It is converting photos into numerical vectors so they can be compared later using distance arithmetic.

---

## Master Registry (Central Database)

A single Python `dict` that lives inside the DSTS coordinator. It is the only global data structure that persists throughout the simulation.

```
master_registry
├── "person_a"  →  { home_building: "b1",  encodings: [ (128,), (128,), ... × 20 ] }
├── "person_b"  →  { home_building: "b1",  encodings: [ (128,), (128,), ... × 20 ] }
├── "person_c"  →  { home_building: "b2",  encodings: [ (128,), (128,), ... × 20 ] }
└── "person_d"  →  { home_building: "b2",  encodings: [ (128,), (128,), ... × 20 ] }
```

Each building also holds a local slice of this — just the encodings for its own registered occupants — so it does not need to scan the full registry on every event.

The DSTS additionally keeps a lightweight parallel lookup (`occupant_id → home_building_id`) for fast routing without loading encodings.

---

## Buildings (BSTS)

Each building is a `BSTS` object. Two are created and registered with the DSTS coordinator.

**b1**
- Zones: `z1_b1`, `z2_b1`, `z3_b1`, `z4_b1`, `zT_b1`
- Registered occupants: `person_a`, `person_b`

**b2**
- Zones: `z1_b2`, `z2_b2`, `z3_b2`, `zT_b2`
- Registered occupants: `person_c`, `person_d`

Each BSTS owns:
- A **registered state table** — fixed rows, one per registered occupant, updated continuously
- A **visitor state table** — starts empty, rows added dynamically when visitors arrive
- An **event log** — ordered list of all `RecognitionEvent` objects fired in this building
- An **event counter** k — integer incremented on every new event, used for ordering

**Clock assumption:** all BSTS clocks are synchronised. The temporal order of any two events across buildings can therefore be directly inferred from their timestamps.

---

## Data Structures

### OccupantEntry

One row in a state table. Represents one person's current probability distribution across all zones in a building.

| Field | Type | Description |
|---|---|---|
| `occupant_id` | `str` | e.g. `"person_a"` |
| `home_building_id` | `str` | The building this person is registered to |
| `probs` | `List[float]` | One probability per zone; last slot is always the transition zone |

**Invariant:** `sum(probs) == 1.0` at all times. Verified after every state update.

Example — `person_a` in b1 (4 internal zones + transition zone):

```
probs index:   0       1       2       3       4
zone label:  z1_b1   z2_b1   z3_b1   z4_b1   zT_b1
```

At initialisation, `probs = [1.0, 0.0, 0.0, 0.0, 0.0]` — person is assumed to be at the entry zone.

---

### StateTable

The full state table for one building — either registered occupants or visitors.

| Field | Type | Description |
|---|---|---|
| `building_id` | `str` | Which building this table belongs to |
| `zone_labels` | `List[str]` | Column headers; last entry is always the transition zone |
| `_rows` | `Dict[str, OccupantEntry]` | Maps `occupant_id → OccupantEntry` |

Two `StateTable` objects exist per building:
- **Registered table** — rows are created at initialisation, never added or removed
- **Visitor table** — starts empty; a row is inserted when a cross-building detection occurs

---

### RecognitionEvent

One event, created each time a CCTV frame is fed into the system.

| Field | Type | Description |
|---|---|---|
| `timestamp` | `float` | Wall-clock time when the image was fed |
| `building_id` | `str` | Which building's camera fired |
| `zone_index` | `int` | Which zone the camera is at (0-indexed) |
| `probs` | `Dict[str, float]` | Probability assigned to each tracked occupant at this event |
| `event_order` | `int` | Sequential counter k within this BSTS |

`probs` has one entry per occupant tracked in this building. The matched person receives the probability derived from their distance score. All others receive a small residual value representing recognition noise.

---

## Recognition Flow (one CCTV frame → one event)

A single test image from `test/` is fed manually, specifying the building and zone. This represents the frame captured by the camera mounted at that zone.

**Step 1 — Face Detection**
`batch_face_locations` uses a CNN detector to find the bounding box of every face in the image.

**Step 2 — Embedding Extraction**
`face_encodings` passes each detected face through the frozen ResNet-34 and returns one 128-d numpy array.

**Step 3 — Distance Scoring (primary decision)**
`face_distance` computes the euclidean distance between the detected embedding and every stored encoding for the building's registered occupants. The person with the lowest distance score is the candidate match.

**Step 4 — Validation (secondary check)**
`compare_faces` applies a fixed threshold of 0.6 to each stored encoding and returns True/False per encoding. If the candidate match accumulates at least **12 True votes** out of their 20 stored encodings, the match is confirmed. This does not override Step 3 — it only validates it.

**Step 5 — Probability Mapping**
The distance score is converted to a probability:

```
p = exp( −5.0 × distance )
```

Lower distance → higher probability. This single value `p` represents how confident the recogniser is that the detected face belongs to the matched person, at this zone, at this moment.

**Step 6 — Visitor Check**
If no match is found among the building's own registered occupants, the detected embedding is compared against all encodings in the master registry. If a match is found there, the person is flagged as a visitor from the other building.

**Step 7 — Event Construction and State Update**
A `RecognitionEvent` is created and the BSTS transition function updates the matched occupant's row in the state table.

---

## Where Zone Probabilities Come From

The camera only fires at **one zone** per event — the zone it is physically mounted at. There is **one image**, **one embedding**, and **one distance score** per event. The model runs exactly once.

The probability `p` from Step 5 tells us how likely this person is to be at **zone j** (the zone where the camera fired).

The probabilities at all **other zones** are not independently measured. They are updated mathematically using the complement formula (Eq. 10):

```
new_prob[l] = (1 − p) × old_prob[l]     for every zone l that is not j
```

So if `p = 0.94` at zone j, then `(1 − p) = 0.06`, and every other zone's old probability is scaled down by 0.06. The intuition is: if we are 94% sure this person is at zone j right now, then we are only 6% as confident in wherever else we thought they might be.

No camera fires at any other zone during this event. The other zone probabilities are purely inferred from the one detection.

---

## What Exactly Changes from One State to the Next

Per event, exactly **one row** in **one state table** is updated — the row belonging to the matched occupant.

Specifically, the `probs` field of that `OccupantEntry` changes. The 5 floats (or 4 floats for b2) get new values based on Eq. 9 and Eq. 10.

Nothing else changes:
- All other occupants' rows are untouched
- The zone labels, building layout, table structure are static
- The master registry is not modified by a normal event
- The visitor table only changes during a visitor arrival

---

## State Update Rules (Eq. 9 and Eq. 10)

For the matched occupant, after an event fires at zone `j` with probability `p`:

**Detected zone j (Eq. 9):**
```
new_prob[j] = p + (1 − p) × old_prob[j]
```
The new probability is the recognition confidence `p` plus whatever residual confidence was already there, scaled by `(1 − p)`.

**All other zones l ≠ j (Eq. 10):**
```
new_prob[l] = (1 − p) × old_prob[l]
```
Each other zone's probability is reduced proportionally.

After both formulas are applied, all values are renormalised to sum to exactly 1.0.

---

## Global State — When It Is Needed and How It Is Stored

Each BSTS continuously maintains its own state table. There is no standing global state table.

The global state (the r-tuple) is **assembled on demand** by iterating over all BSTS objects, flattening their current state tables, and concatenating the results into a single Python `tuple` of floats. This tuple is temporary — it is computed, used, and discarded. It is never stored between queries.

**The global state is assembled only when:**
- A **spatio-temporal query** is issued — locating an occupant, generating a movement track, answering "who is in building X right now"
- A **visitor arrival** requires checking the occupant's last known state across buildings before updating the visitor table

For all normal within-building recognition events, only the local BSTS state table is touched. The global state is never involved.

---

## Visitor Arrival Flow

When `person_a` (registered in b1) is detected at b2's entry zone:

1. b2's local recogniser checks against `person_c` and `person_d` — no match
2. Escalates to master registry — matches `person_a`
3. A new row for `person_a` is added to b2's visitor state table, initialised with `1.0` at zone 0
4. b1's registered table is updated: `person_a`'s transition zone (`zT_b1`) probability is set to `0.98`, confirming they have left b1
5. The global state, if queried now, will reflect `person_a` in b2's visitor table and their high transition zone probability in b1

---

## Global Event Log

All events across both buildings are collected into a single ordered list in the DSTS coordinator. Sorted by:

1. Timestamp — primary (clocks are synchronised so this is reliable across buildings)
2. Building ID alphabetically — b1 before b2 on an exact tie
3. Event counter k within the BSTS — tertiary

---

## Quick Reference

| Name | Type | What it holds |
|---|---|---|
| `master_registry` | `Dict[str, Dict]` | Persistent central DB: person → home building + 20 stored embeddings |
| `OccupantEntry.probs` | `List[float]` | Current probability vector across all zones for one person |
| `StateTable._rows` | `Dict[str, OccupantEntry]` | All rows in one state table (registered or visitor) |
| `RecognitionEvent.probs` | `Dict[str, float]` | Per-person recognition probabilities assigned at one event |
| `DSTS.global_event_log` | `List[RecognitionEvent]` | All events across both buildings, timestamp-ordered |
| Global state (r-tuple) | `tuple` of `float` | Temporary, assembled on demand, never stored persistently |
| Face encoding | `numpy.ndarray` shape `(128,)` | 128-d face embedding from frozen ResNet-34 |