"""
DSTS Implementation - Multi-Building Surveillance System
Based on: Mohan & Menon, "Modelling large scale camera networks for
identification and tracking: an abstract framework", IET Computer Vision 2020
"""

import face_recognition
import numpy as np
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math
import copy

# ============================================================================
# CONSTANTS (Paper Section 4)
# ============================================================================
IMAGES_PER_PERSON = 40
MIN_VOTE_THRESHOLD = 12
FACE_MATCH_TOLERANCE = 0.6
DISTANCE_DECAY_FACTOR = 5.0
NUM_BUILDINGS = 5
OCCUPANTS_PER_BUILDING = 5
TOTAL_OCCUPANTS = NUM_BUILDINGS * OCCUPANTS_PER_BUILDING
PROBABILITY_SUM_TOLERANCE = 1e-9
NUM_EVENTS = 86

BUILDING_ZONE_CONFIGS = {
    0: ['z1_b0', 'z2_b0', 'z3_b0', 'z4_b0', 'zT_b0'],
    1: ['z1_b1', 'z2_b1', 'z3_b1', 'z4_b1', 'zT_b1'],
    2: ['z1_b2', 'z2_b2', 'z3_b2', 'z4_b2', 'zT_b2'],
    3: ['z1_b3', 'z2_b3', 'z3_b3', 'z4_b3', 'zT_b3'],
    4: ['z1_b4', 'z2_b4', 'z3_b4', 'z4_b4', 'zT_b4'],
}

# ============================================================================
# DATA STRUCTURES (Paper Section 3.1)
# ============================================================================

@dataclass
class OccupantEntry:
    """
    One row in a state table (Table 1 from paper).
    Paper Eq. 2: probability distribution across zones.
    """
    occupant_id: str
    probs: Dict[str, float]

    def verify_constraint(self) -> bool:
        """Verify Equation 6: sum of probs = 1.0"""
        total = sum(self.probs.values())
        is_valid = abs(total - 1.0) < PROBABILITY_SUM_TOLERANCE
        if not is_valid:
            print(f"WARNING: Constraint violation for {self.occupant_id}: sum = {total:.10f}")
        return is_valid

    def renormalize(self) -> None:
        """Renormalize probabilities to sum to 1.0"""
        total = sum(self.probs.values())
        if total > 0:
            self.probs = {zone: prob / total for zone, prob in self.probs.items()}


@dataclass
class StateTable:
    """
    Registered State Table (Table 1 from paper).
    Paper Section 3.2: Fixed rows for building's registered occupants.
    """
    entries: Dict[str, OccupantEntry] = field(default_factory=dict)

    def add_occupant(self, occupant_id: str, zones: List[str],
                     initial_zone: Optional[str] = None) -> None:
        if initial_zone and initial_zone in zones:
            probs = {zone: 0.0 for zone in zones}
            probs[initial_zone] = 1.0
        else:
            prob_per_zone = 1.0 / len(zones)
            probs = {zone: prob_per_zone for zone in zones}
        entry = OccupantEntry(occupant_id=occupant_id, probs=probs)
        entry.verify_constraint()
        self.entries[occupant_id] = entry

    def get_occupant(self, occupant_id: str) -> Optional[OccupantEntry]:
        return self.entries.get(occupant_id)

    def get_all_occupants(self) -> List[str]:
        return list(self.entries.keys())


@dataclass
class RecognitionEvent:
    """
    Paper Section 3.1: e_k^u = <t, z_j^u, P_k>
    """
    timestamp: datetime
    building_id: int
    zone_id: str
    event_counter: int
    probabilities: Dict[str, float]
    matched_occupant: str

    def __repr__(self) -> str:
        return (f"Event(t={self.timestamp.strftime('%H:%M:%S')}, "
                f"b{self.building_id}:{self.zone_id}, "
                f"{self.matched_occupant}, "
                f"p={self.probabilities.get(self.matched_occupant, 0.0):.3f})")


# ============================================================================
# BSTS - Building-Specific State Transition System (Paper Section 3.1)
# ============================================================================

class BSTS:
    """BSTS_u = <S^u, E^u, Delta^u>"""

    def __init__(self, building_id: int, zones: List[str],
                 registered_occupants: List[str]):
        self.building_id = building_id
        self.zones = zones
        self.num_zones = len(zones)
        self.registered_occupants = set(registered_occupants)
        self.registered_table = StateTable()
        for occupant_id in registered_occupants:
            self.registered_table.add_occupant(occupant_id, zones, initial_zone=zones[0])
        self.event_counter = 0
        self.event_history: List[RecognitionEvent] = []
        self.state_counter = 0

    def is_registered(self, occupant_id: str) -> bool:
        return occupant_id in self.registered_occupants

    def get_occupant_entry(self, occupant_id: str) -> Optional[OccupantEntry]:
        return self.registered_table.get_occupant(occupant_id)

    def apply_transition(self, event: RecognitionEvent) -> None:
        """Equations 9 and 10: state transition function Delta^u"""
        occupant_id = event.matched_occupant
        detected_zone = event.zone_id
        recognition_prob = event.probabilities[occupant_id]
        table = self.registered_table
        entry = table.get_occupant(occupant_id)
        if entry is None:
            return
        current_probs = copy.deepcopy(entry.probs)
        x_i = 1.0 - recognition_prob
        new_probs = {}
        for zone in self.zones:
            if zone == detected_zone:
                new_probs[zone] = recognition_prob + x_i * current_probs[zone]
            else:
                new_probs[zone] = x_i * current_probs[zone]
        entry.probs = new_probs
        entry.renormalize()
        entry.verify_constraint()
        self.state_counter += 1
        self.event_history.append(event)

    def get_current_state(self) -> StateTable:
        """Paper Eq. 3: current state"""
        return self.registered_table

    def query_occupant_location(self, occupant_id: str,
                                theta: float = 0.5) -> Optional[Tuple[str, float]]:
        entry = self.get_occupant_entry(occupant_id)
        if entry is None:
            return None
        max_zone = max(entry.probs.items(), key=lambda x: x[1])
        zone_id, prob = max_zone
        if prob >= theta:
            return (zone_id, prob)
        return None


# ============================================================================
# FACE RECOGNITION ENGINE (Paper Section 4)
# ============================================================================

def distance_to_probability(distance: float) -> float:
    """p = exp(-5.0 * d)"""
    probability = math.exp(-DISTANCE_DECAY_FACTOR * distance)
    return max(0.0, min(1.0, probability))


class FaceRecognitionEngine:
    def __init__(self, encodings_database: Dict[str, List[np.ndarray]]):
        self.encodings_database = encodings_database

    def recognize_from_encoding(self, captured_encoding: np.ndarray,
                                candidate_occupants: List[str]) -> Tuple[str, float, int]:
        min_distance = float('inf')
        matched_id = None
        vote_count = 0
        all_votes = []
        for occupant_id in candidate_occupants:
            if occupant_id not in self.encodings_database:
                continue
            occupant_encodings = self.encodings_database[occupant_id]
            for encoding in occupant_encodings:
                distance = face_recognition.face_distance([encoding], captured_encoding)[0]
                is_match = face_recognition.compare_faces([encoding], captured_encoding,
                                                          tolerance=FACE_MATCH_TOLERANCE)[0]
                if is_match:
                    all_votes.append(occupant_id)
                if occupant_id == matched_id or matched_id is None:
                    if distance < min_distance:
                        min_distance = distance
                        matched_id = occupant_id
        if matched_id:
            vote_count = all_votes.count(matched_id)
        probability = distance_to_probability(min_distance) if matched_id else 0.0
        return (matched_id, probability, vote_count)

    def recognize_at_building(self, captured_encoding: np.ndarray,
                              building: BSTS) -> Tuple[Optional[str], float]:
        registered_occupants = list(building.registered_occupants)
        matched_id, probability, vote_count = self.recognize_from_encoding(
            captured_encoding, registered_occupants)
        if matched_id and vote_count >= MIN_VOTE_THRESHOLD:
            return (matched_id, probability)
        return (None, 0.0)


# ============================================================================
# DATA LOADING WITH GAUSSIAN AUGMENTATION
# ============================================================================

def augment_encodings(encodings: List[np.ndarray], target_count: int) -> List[np.ndarray]:
    """Fill missing encodings using Gaussian noise on existing ones."""
    augmented = list(encodings)
    while len(augmented) < target_count:
        base = encodings[np.random.randint(0, len(encodings))]
        noise = np.random.normal(0, 0.01, base.shape)
        augmented.append(base + noise)
    return augmented[:target_count]


def load_lfw_data_and_generate_encodings(min_faces: int = 15) -> Tuple:
    print(f"Loading LFW dataset (min {min_faces} images/person)...")
    lfw_data = fetch_lfw_people(min_faces_per_person=min_faces, resize=0.4, color=True)
    selected_people = []
    encodings_database = {}
    for person_name in lfw_data.target_names:
        if len(selected_people) >= TOTAL_OCCUPANTS:
            break
        person_idx = np.where(lfw_data.target_names == person_name)[0][0]
        image_indices = np.where(lfw_data.target == person_idx)[0]
        encodings_list = []
        for idx in image_indices[:IMAGES_PER_PERSON]:
            img_uint8 = (lfw_data.images[idx] * 255).astype(np.uint8)
            encodings = face_recognition.face_encodings(img_uint8)
            if encodings:
                encodings_list.append(encodings[0])
        if len(encodings_list) > 0:
            # Augment with Gaussian noise if fewer than IMAGES_PER_PERSON
            if len(encodings_list) < IMAGES_PER_PERSON:
                encodings_list = augment_encodings(encodings_list, IMAGES_PER_PERSON)
            selected_people.append(person_name)
            encodings_database[person_name] = encodings_list
            print(f"  {len(selected_people):2d}. {person_name} ({len(encodings_list)} encodings)")
    if len(selected_people) < TOTAL_OCCUPANTS:
        raise ValueError(f"Found {len(selected_people)}, need {TOTAL_OCCUPANTS}. Lower min_faces.")
    return lfw_data, selected_people, encodings_database


def partition_occupants(selected_people: List[str]) -> Dict[int, List[str]]:
    assignments = {}
    for bid in range(NUM_BUILDINGS):
        start = bid * OCCUPANTS_PER_BUILDING
        end = start + OCCUPANTS_PER_BUILDING
        assignments[bid] = selected_people[start:end]
        print(f"Building {bid}: {assignments[bid]}")
    return assignments


def create_central_directory(assignments: Dict[int, List[str]],
                             encodings_db: Dict[str, List[np.ndarray]]) -> Dict[str, Dict]:
    directory = {}
    for bid, occupants in assignments.items():
        for oid in occupants:
            if oid in encodings_db:
                directory[oid] = {'home_building': bid, 'encodings': encodings_db[oid]}
    print(f"\nCentral directory: {len(directory)} occupants.")
    return directory


# ============================================================================
# DSTS - Distributed State Transition System (Paper Section 3.1)
# ============================================================================

class DSTS:
    """DSTS = <S', E', Delta'>"""

    def __init__(self):
        self.buildings: Dict[int, BSTS] = {}
        self.recognition_engine: Optional[FaceRecognitionEngine] = None
        self.global_event_log: List[RecognitionEvent] = []
        self.occupant_registry: Dict[str, int] = {}
        self.current_time = datetime.now().replace(hour=9, minute=0, second=0)

    def register_building(self, bsts: BSTS) -> None:
        self.buildings[bsts.building_id] = bsts
        for oid in bsts.registered_occupants:
            self.occupant_registry[oid] = bsts.building_id
        print(f"Registered Building {bsts.building_id}: {len(bsts.registered_occupants)} occupants")

    def set_recognition_engine(self, engine: FaceRecognitionEngine) -> None:
        self.recognition_engine = engine

    def happened_before(self, e1: RecognitionEvent, e2: RecognitionEvent) -> bool:
        """Paper Section 3.3, Equation 12"""
        if e1.building_id == e2.building_id:
            return e1.event_counter < e2.event_counter
        if e1.timestamp == e2.timestamp:
            return e1.building_id < e2.building_id
        return e1.timestamp < e2.timestamp

    def are_concurrent(self, e1: RecognitionEvent, e2: RecognitionEvent) -> bool:
        if e1.building_id == e2.building_id:
            return False
        return e1.timestamp == e2.timestamp

    def _insert_ordered(self, event: RecognitionEvent) -> None:
        left, right = 0, len(self.global_event_log)
        while left < right:
            mid = (left + right) // 2
            if self.happened_before(event, self.global_event_log[mid]):
                right = mid
            else:
                left = mid + 1
        self.global_event_log.insert(left, event)

    def submit_recognition(self, building_id: int, zone_id: str,
                           captured_encoding: np.ndarray,
                           timestamp: Optional[datetime] = None) -> RecognitionEvent:
        if self.recognition_engine is None:
            raise ValueError("Recognition engine not set")
        building = self.buildings[building_id]
        if timestamp is None:
            timestamp = self.current_time
        matched_id, probability = self.recognition_engine.recognize_at_building(
            captured_encoding, building)
        if matched_id is None:
            print(f"Unknown individual at Building {building_id}, {zone_id}")
            return RecognitionEvent(timestamp=timestamp, building_id=building_id,
                                   zone_id=zone_id, event_counter=-1,
                                   probabilities={}, matched_occupant="UNKNOWN")
        probabilities = {matched_id: probability}
        all_tracked = set(building.registered_table.get_all_occupants())
        remaining = 1.0 - probability
        others = [o for o in all_tracked if o != matched_id]
        if others:
            residual = remaining / len(others)
            for o in others:
                probabilities[o] = residual
        building.event_counter += 1
        event = RecognitionEvent(timestamp=timestamp, building_id=building_id,
                                 zone_id=zone_id, event_counter=building.event_counter,
                                 probabilities=probabilities, matched_occupant=matched_id)
        building.apply_transition(event)
        self._insert_ordered(event)
        return event

    def get_global_state(self, at_time: Optional[datetime] = None) -> Dict[int, StateTable]:
        """Equations 4 & 5"""
        global_state = {}
        for bid, bld in self.buildings.items():
            global_state[bid] = bld.get_current_state()
        return global_state

    def query_occupant_location(self, occupant_id: str,
                                theta: float = 0.5) -> Optional[Tuple[int, str, float]]:
        max_prob = 0.0
        max_loc = None
        for bid, bld in self.buildings.items():
            result = bld.query_occupant_location(occupant_id, theta)
            if result:
                zid, prob = result
                if prob > max_prob:
                    max_prob = prob
                    max_loc = (bid, zid, prob)
        return max_loc

    def generate_occupant_track(self, occupant_id: str) -> List[Tuple]:
        track = []
        for event in self.global_event_log:
            if event.matched_occupant == occupant_id:
                prob = event.probabilities.get(occupant_id, 0.0)
                track.append((event.timestamp, event.building_id, event.zone_id, prob))
        return track

    def advance_time(self, minutes: int) -> None:
        self.current_time += timedelta(minutes=minutes)


# ============================================================================
# PERFORMANCE EVALUATOR (Paper Section 4, Tables 5 & 6)
# ============================================================================

class PerformanceEvaluator:
    def __init__(self, dsts: DSTS):
        self.dsts = dsts

    def evaluate_state(self, building_id: int, ground_truth: Dict[str, str],
                       theta: float) -> Dict[str, float]:
        building = self.dsts.buildings[building_id]
        recognized = {}
        for oid in building.registered_table.get_all_occupants():
            entry = building.registered_table.get_occupant(oid)
            if entry:
                max_zone = max(entry.probs.items(), key=lambda x: x[1])
                zid, prob = max_zone
                if prob >= theta:
                    recognized[oid] = zid
        tp = sum(1 for o in recognized if o in ground_truth)
        fp = sum(1 for o in recognized if o not in ground_truth)
        fn = sum(1 for o in ground_truth if o not in recognized)
        ak, bk, ck = tp, tp + fp, tp + fn
        precision = ak / bk if bk > 0 else 0.0
        recall = ak / ck if ck > 0 else 0.0
        return {'tp': tp, 'fp': fp, 'fn': fn, 'ak': ak, 'bk': bk, 'ck': ck,
                'precision': precision, 'recall': recall}

    def evaluate_average_metrics(self, event_history: List[Tuple],
                                  theta_values: List[float]) -> Dict[float, Dict]:
        results = {}
        for theta in theta_values:
            precs, recs = [], []
            for bid, gt in event_history:
                m = self.evaluate_state(bid, gt, theta)
                precs.append(m['precision'])
                recs.append(m['recall'])
            results[theta] = {'avg_precision': np.mean(precs) if precs else 0.0,
                              'avg_recall': np.mean(recs) if recs else 0.0}
        return results

    def find_optimal_threshold(self, event_history: List[Tuple],
                                num_points: int = 20) -> float:
        thetas = np.linspace(0.0, 1.0, num_points)
        results = self.evaluate_average_metrics(event_history, thetas.tolist())
        min_diff, optimal = float('inf'), 0.5
        for t, m in results.items():
            d = abs(m['avg_precision'] - m['avg_recall'])
            if d < min_diff:
                min_diff, optimal = d, t
        return optimal


# ============================================================================
# SIMULATION
# ============================================================================

def simulate_movement(dsts: DSTS, encodings_db: Dict[str, List[np.ndarray]],
                      num_events: int = NUM_EVENTS) -> List[Tuple]:
    event_history = []
    ground_truth = {}
    active = []
    for bid, bld in dsts.buildings.items():
        olist = list(bld.registered_occupants)[:2]
        active.extend([(o, bid) for o in olist])
    print(f"\nSimulating {num_events} events...")
    print(f"Active: {[o for o, _ in active]}\n")
    states = {oid: (hbid, None, -1) for oid, hbid in active}
    count = 0
    while count < num_events:
        oid, hbid = active[np.random.randint(0, len(active))]
        cbid, czone, zidx = states[oid]
        bld = dsts.buildings[cbid]
        zones = bld.zones[:-1]
        if czone is None:
            nzidx, nzone, action = 0, zones[0], "ENTRY"
        elif zidx < len(zones) - 1:
            nzidx = zidx + 1
            nzone, action = zones[nzidx], "MOVE"
        else:
            nzone = bld.zones[-1]
            nzidx, action = -1, "EXIT"
        encs = encodings_db[oid]
        cap = encs[np.random.randint(0, len(encs))]
        cap = cap + np.random.normal(0, 0.01, cap.shape)
        event = dsts.submit_recognition(cbid, nzone, cap, dsts.current_time)
        if action == "EXIT":
            ground_truth.pop(oid, None)
            states[oid] = (hbid, None, -1)
        else:
            ground_truth[oid] = nzone
            states[oid] = (cbid, nzone, nzidx)
        event_history.append((cbid, copy.deepcopy(ground_truth)))
        dsts.advance_time(np.random.randint(1, 6))
        count += 1
        if count % 10 == 0:
            print(f"  {count}/{num_events} events")
        if count <= 5:
            print(f"    {action}: {oid} @ B{cbid}, {nzone} "
                  f"(p={event.probabilities.get(oid, 0.0):.3f})")
    print(f"\nSimulation complete: {count} events\n")
    return event_history


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_registered(assignments, directory):
    print("=" * 80)
    print("REGISTERED OCCUPANTS BY BUILDING")
    print("=" * 80 + "\n")
    for bid in sorted(assignments.keys()):
        occs = assignments[bid]
        print(f"Building {bid}:")
        for i, oid in enumerate(occs, 1):
            ne = len(directory[oid]['encodings'])
            print(f"  {i}. {oid} [{ne} encodings]")
        print()


def display_state_tables(dsts, bid, theta=0.5):
    bld = dsts.buildings[bid]
    zones = bld.zones
    print(f"\n{'=' * 100}")
    print(f"STATE TABLE FOR BUILDING {bid}")
    print(f"{'=' * 100}\n")
    header = f"{'Occupant':<25}"
    for z in zones:
        header += f"{z:>12}"
    print(header)
    print("-" * 100)
    for oid in sorted(bld.registered_table.get_all_occupants()):
        entry = bld.registered_table.get_occupant(oid)
        if entry:
            row = f"{oid:<25}"
            for z in zones:
                p = entry.probs[z]
                row += f"{p:>12.3f}{'*' if p >= theta else ' '}"
            print(row)
    print("-" * 100)
    print(f"* >= theta={theta}\n")


def display_track(dsts, oid):
    track = dsts.generate_occupant_track(oid)
    if not track:
        print(f"No tracking data for {oid}")
        return
    print(f"\n{'=' * 80}")
    print(f"TRACK: {oid}")
    print(f"{'=' * 80}\n")
    print(f"{'Time':<15} {'Bldg':<10} {'Zone':<12} {'Prob':<10}")
    print("-" * 80)
    for ts, bid, zid, prob in track:
        print(f"{ts.strftime('%H:%M:%S'):<15} B{bid:<9} {zid:<12} {prob:<10.3f}")
    print(f"-" * 80 + f"\nTotal: {len(track)}\n")


def display_pr_table(metrics, title="PRECISION-RECALL"):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}\n")
    print(f"{'θ':<12} {'Avg Precision':<18} {'Avg Recall':<18}")
    print("-" * 80)
    for t in sorted(metrics.keys()):
        p, r = metrics[t]['avg_precision'], metrics[t]['avg_recall']
        mark = " ← OPTIMAL" if abs(p - r) < 0.05 else ""
        print(f"{t:<12.2f} {p:<18.3f} {r:<18.3f}{mark}")
    print("-" * 80 + "\n")


def plot_pr_curve(metrics, save_path=None):
    thetas = sorted(metrics.keys())
    precs = [metrics[t]['avg_precision'] for t in thetas]
    recs = [metrics[t]['avg_recall'] for t in thetas]
    plt.figure(figsize=(10, 6))
    plt.plot(thetas, precs, 'b-o', label='Avg Precision', linewidth=2)
    plt.plot(thetas, recs, 'r-s', label='Avg Recall', linewidth=2)
    idx = np.argmin([abs(p - r) for p, r in zip(precs, recs)])
    plt.axvline(x=thetas[idx], color='green', linestyle='--',
                label=f'Optimal θ = {thetas[idx]:.2f}')
    plt.xlabel('θ', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Average Precision and Recall vs θ', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1.1)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    plt.close()


def display_event_log(dsts, n=10):
    print(f"\n{'=' * 100}")
    print(f"GLOBAL EVENT LOG (Last {n})")
    print(f"{'=' * 100}\n")
    print(f"{'#':<5} {'Time':<15} {'Bldg':<6} {'Zone':<12} {'Occupant':<25} {'Prob':<8}")
    print("-" * 100)
    for i, ev in enumerate(dsts.global_event_log[-n:], 1):
        p = ev.probabilities.get(ev.matched_occupant, 0.0)
        print(f"{i:<5} {ev.timestamp.strftime('%H:%M:%S'):<15} B{ev.building_id:<5} "
              f"{ev.zone_id:<12} {ev.matched_occupant:<25} {p:<8.3f}")
    print(f"-" * 100 + f"\nTotal: {len(dsts.global_event_log)}\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("DSTS SIMULATION - Mohan & Menon, IET Computer Vision 2020")
    print("=" * 80 + "\n")

    # Load data
    print("STEP 1: Loading LFW and generating encodings...")
    lfw_data, people, enc_db = load_lfw_data_and_generate_encodings()

    # Partition
    print("\nSTEP 2: Partitioning occupants...")
    assignments = partition_occupants(people)

    # Central directory
    print("\nSTEP 3: Central directory...")
    directory = create_central_directory(assignments, enc_db)

    # Display
    display_registered(assignments, directory)

    # Create BSTS
    print("STEP 4: Creating BSTS instances...")
    buildings = {}
    for bid, occs in assignments.items():
        zones = BUILDING_ZONE_CONFIGS[bid]
        bsts = BSTS(bid, zones, occs)
        buildings[bid] = bsts
        print(f"  BSTS {bid}: {len(zones)} zones, {len(occs)} occupants")

    # DSTS
    print("\nSTEP 5: Initializing DSTS...")
    dsts = DSTS()
    for bid, bsts in buildings.items():
        dsts.register_building(bsts)
    engine = FaceRecognitionEngine(enc_db)
    dsts.set_recognition_engine(engine)
    print("DSTS ready!")

    # Simulate
    print("\nSTEP 6: Simulating movement...")
    history = simulate_movement(dsts, enc_db, NUM_EVENTS)

    # State tables
    print("\nSTEP 7: State tables...")
    for bid in range(NUM_BUILDINGS):
        display_state_tables(dsts, bid)

    # Tracks
    print("\nSTEP 8: Tracks...")
    active = []
    for bid, bld in dsts.buildings.items():
        active.extend(list(bld.registered_occupants)[:2])
    for oid in active[:3]:
        display_track(dsts, oid)

    # Precision-recall
    print("\nSTEP 9: Precision-recall...")
    ev = PerformanceEvaluator(dsts)
    thetas = [0.0, 0.1, 0.3, 0.5, 0.82, 0.9, 1.0]
    metrics = ev.evaluate_average_metrics(history, thetas)
    display_pr_table(metrics, "AVERAGE PRECISION AND RECALL (Paper Table 6)")
    opt = ev.find_optimal_threshold(history)
    print(f"Optimal θ = {opt:.3f} (paper reports 0.82)\n")

    # Plot
    print("STEP 10: Plotting...")
    detailed = ev.evaluate_average_metrics(history, np.linspace(0, 1, 20).tolist())
    plot_pr_curve(detailed, 'precision_recall_curve.png')

    # Event log
    print("\nSTEP 11: Event log...")
    display_event_log(dsts, 15)

    # Queries
    print("\nSTEP 12: Location queries (θ=0.5)...")
    print("=" * 80)
    for oid in active[:5]:
        r = dsts.query_occupant_location(oid, 0.5)
        if r:
            bid, zid, prob = r
            print(f"{oid:<30} → B{bid}, {zid} (p={prob:.3f})")
        else:
            print(f"{oid:<30} → Unknown (p < θ)")

    # Summary
    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"  Buildings: {len(dsts.buildings)}")
    print(f"  Occupants: {sum(len(b.registered_occupants) for b in dsts.buildings.values())}")
    print(f"  Events: {len(dsts.global_event_log)}")
    print(f"  Time: {dsts.current_time.strftime('%H:%M:%S')}\n")


if __name__ == "__main__":
    main()
