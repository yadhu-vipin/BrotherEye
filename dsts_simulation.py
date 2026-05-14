"""
DSTS Implementation - Multi-Building Surveillance System
Based on: Mohan & Menon, "Modelling large scale camera networks for
identification and tracking: an abstract framework", IET Computer Vision 2020
"""
import os
# Allow a runtime override to force mock recognition even if face_recognition is importable.
_FORCE_MOCK = os.environ.get('BROTHEREYE_FORCE_MOCK', '').lower() in ('1', 'true', 'yes')
if _FORCE_MOCK:
    face_recognition = None
    FACE_RECOG_AVAILABLE = False
    print("BROTHEREYE_FORCE_MOCK set -> forcing mock recognition mode.")
else:
    try:
        # attempt to import the optional heavy dependency
        import face_recognition
        FACE_RECOG_AVAILABLE = True
    except Exception:
        face_recognition = None
        FACE_RECOG_AVAILABLE = False
        print("NOTE: 'face_recognition' library not available. Running in mock recognition mode.")

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import math
import copy
import heapq
from sklearn.datasets import fetch_lfw_people  # noqa: E402
import matplotlib.pyplot as plt

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
# SPATIO-TEMPORAL ZONE GRAPH
# ============================================================================

class ZoneGraph:
    """Simple graph of zones with travel times (seconds) on edges.

    It supports shortest-path queries between zone IDs using Dijkstra.
    Default construction connects zones in BUILDING_ZONE_CONFIGS sequentially
    and connects all transition zones (`zT_*`) between buildings with a
    configurable inter-building travel time.
    """

    def __init__(self, building_zone_configs: Dict[int, List[str]],
                 intra_zone_time: float = 30.0,
                 inter_building_time: float = 300.0):
        # adjacency: zone_id -> list of (neighbor_zone_id, travel_time_seconds)
        self.adj: Dict[str, List[Tuple[str, float]]] = {}
        self.building_zone_configs = building_zone_configs
        self.intra_zone_time = intra_zone_time
        self.inter_building_time = inter_building_time
        self._build_default_graph()

    def _add_edge(self, a: str, b: str, t: float) -> None:
        self.adj.setdefault(a, []).append((b, t))
        self.adj.setdefault(b, []).append((a, t))

    def _build_default_graph(self) -> None:
        # connect sequential zones within each building
        for bid, zones in self.building_zone_configs.items():
            for i in range(len(zones) - 1):
                a = zones[i]
                b = zones[i + 1]
                self._add_edge(a, b, self.intra_zone_time)
        # connect transition zones across buildings
        # find all zT nodes
        transition_nodes = []
        for bid, zones in self.building_zone_configs.items():
            for z in zones:
                if z.startswith('zT_'):
                    transition_nodes.append(z)
        for i in range(len(transition_nodes)):
            for j in range(i + 1, len(transition_nodes)):
                self._add_edge(transition_nodes[i], transition_nodes[j], self.inter_building_time)

    def shortest_travel_time(self, src: str, dst: str) -> Optional[float]:
        """Return shortest travel time in seconds between src and dst, or None if unreachable."""
        if src == dst:
            return 0.0
        if src not in self.adj or dst not in self.adj:
            return None
        # Dijkstra
        pq = [(0.0, src)]
        dist = {src: 0.0}
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, float('inf')):
                continue
            if u == dst:
                return d
            for v, w in self.adj.get(u, []):
                nd = d + w
                if nd < dist.get(v, float('inf')):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return None


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
                if FACE_RECOG_AVAILABLE:
                    distance = face_recognition.face_distance([encoding], captured_encoding)[0]
                    is_match = face_recognition.compare_faces([encoding], captured_encoding,
                                                              tolerance=FACE_MATCH_TOLERANCE)[0]
                else:
                    # Fallback: use Euclidean distance on the encoding vectors and a simple
                    # threshold comparator to simulate compare_faces behaviour.
                    distance = float(np.linalg.norm(encoding - captured_encoding))
                    is_match = distance <= FACE_MATCH_TOLERANCE
                if is_match:
                    all_votes.append(occupant_id)
                # Always update the global minimum distance found so far.
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
        if FACE_RECOG_AVAILABLE:
            if matched_id and vote_count >= MIN_VOTE_THRESHOLD:
                return (matched_id, probability)
        else:
            # mock mode: accept nearest match directly
            if matched_id:
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
            if FACE_RECOG_AVAILABLE and face_recognition is not None:
                encs = face_recognition.face_encodings(img_uint8)
                if encs:
                    encodings_list.append(encs[0])
            else:
                # deterministic pseudo-encoding: project image pixels to a fixed-length vector
                # using a reproducible hash + simple downsampling/normalization. This is
                # only used for mock/demo/testing purposes when face_recognition isn't
                # available.
                flat = img_uint8.flatten().astype(np.float32)
                # take a deterministic slice/stride to build a vector of length 128
                if flat.size < 128:
                    vec = np.pad(flat, (0, 128 - flat.size))[:128]
                else:
                    stride = max(1, flat.size // 128)
                    vec = flat[::stride][:128]
                # normalize
                vec = (vec - np.mean(vec)) / (np.std(vec) + 1e-6)
                encodings_list.append(vec)
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
        # spatio-temporal zone graph used for reachability/travel-time reasoning
        self.zone_graph = ZoneGraph(BUILDING_ZONE_CONFIGS)


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
        # First attempt: direct high-confidence zones from BSTS
        max_prob = 0.0
        max_loc = None
        for bid, bld in self.buildings.items():
            result = bld.query_occupant_location(occupant_id, theta)
            if result:
                zid, prob = result
                if prob > max_prob:
                    max_prob = prob
                    max_loc = (bid, zid, prob)
        if max_loc:
            return max_loc

        # If no direct confident result, fall back to spatio-temporal inference
        inferred = self.infer_possible_locations(occupant_id, at_time=self.current_time)
        if not inferred:
            return None
        # pick the max-probability inferred location
        best_zone, best_prob = max(inferred.items(), key=lambda x: x[1])
        # best_zone is like 'z1_b0' — need to map back to building id
        # find building id by checking BUILDING_ZONE_CONFIGS
        for bid, zones in BUILDING_ZONE_CONFIGS.items():
            if best_zone in zones:
                return (bid, best_zone, best_prob)
        # should not reach here, but return None if mapping fails
        return None

    def infer_possible_locations(self, occupant_id: str, at_time: Optional[datetime] = None,
                                 mobility_tau: float = 1.0) -> Dict[str, float]:
        """Infer a probability distribution over zones for occupant_id at at_time.

        Simple algorithm:
        - Find the last event for the occupant (most recent in global_event_log)
        - Use the BSTS state distribution for that time as source distribution
        - For each source zone, compute shortest travel time to every zone
          and allow contribution only if travel_time <= delta_t * mobility_tau
        - Score contribution = source_prob * exp(-beta * max(0, travel_time - delta_t))
        - Renormalize and return mapping zone->probability
        """
        if at_time is None:
            at_time = self.current_time
        # find last event for this occupant in global_event_log
        last_event = None
        for ev in reversed(self.global_event_log):
            if ev.matched_occupant == occupant_id:
                last_event = ev
                break
        # If no event found, return empty
        if last_event is None:
            return {}

        delta = (at_time - last_event.timestamp).total_seconds()
        if delta < 0:
            delta = 0.0

        # build source distribution: use the entry probs from the building's state table
        src_building = last_event.building_id
        bld = self.buildings.get(src_building)
        if bld is None:
            return {}
        entry = bld.get_occupant_entry(occupant_id)
        if entry is None:
            return {}

        beta = 0.01
        contributions: Dict[str, float] = {}
        for src_zone, src_prob in entry.probs.items():
            if src_prob <= 0:
                continue
            # for each candidate zone, compute travel time
            for bid, zones in BUILDING_ZONE_CONFIGS.items():
                for dst_zone in zones:
                    tt = self.zone_graph.shortest_travel_time(src_zone, dst_zone)
                    if tt is None:
                        continue
                    # allow some slack via mobility_tau
                    if tt <= delta * mobility_tau + 1e-6:
                        # score decays with unused time gap: if travel_time much smaller than delta, prefer closer
                        gap = max(0.0, tt - delta)
                        score = src_prob * math.exp(-beta * gap)
                        contributions[dst_zone] = contributions.get(dst_zone, 0.0) + score

        # renormalize
        total = sum(contributions.values())
        if total <= 0:
            return {}
        return {z: p / total for z, p in contributions.items()}

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
    print(f"{'theta':<12} {'Avg Precision':<18} {'Avg Recall':<18}")
    print("-" * 80)
    for t in sorted(metrics.keys()):
        p, r = metrics[t]['avg_precision'], metrics[t]['avg_recall']
        mark = " ← OPTIMAL" if abs(p - r) < 0.05 else ""
        print(f"{t:<12.2f} {p:<18.3f} {r:<18.3f}{mark}")
    print("-" * 80 + "\n")


def plot_pr_curve(metrics, save_path=None, title=None):
    thetas = sorted(metrics.keys())
    precs = [metrics[t]['avg_precision'] for t in thetas]
    recs = [metrics[t]['avg_recall'] for t in thetas]
    
    plt.figure(figsize=(10, 8))
    plt.plot(thetas, precs, color='red', linestyle='-', label='Avg Prec', linewidth=2)
    plt.plot(thetas, recs, color='yellow', linestyle='--', label='Avg Rec', linewidth=2)
    
    if title is None:
        title = r'Average Precision and Average Recall vs Recognition Threshold  $\theta$'
    
    plt.title(title, fontsize=14)
    plt.xlabel(r'Recognition Threshold  $\theta$', fontsize=12)
    
    plt.legend(loc='lower left', fontsize=11, frameon=True)
    plt.grid(True, linestyle='-', alpha=0.3)
    
    plt.xlim(0.0, 1.1)
    plt.ylim(0.0, 1.1)
    
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
    print(f"Optimal theta = {opt:.3f} (paper reports 0.82)\n")

    # Plot Average Metrics (Fig 5)
    print("STEP 10.1: Plotting average metrics (Fig 5)...")
    detailed = ev.evaluate_average_metrics(history, np.linspace(0, 1.1, 50).tolist())
    plot_pr_curve(detailed, 'average_pr_curve.png')

    # Plot State-based Metrics (Fig 4)
    print("STEP 10.2: Plotting state-based metrics (Fig 4)...")
    if len(history) >= 30:
        bid, gt = history[29]
        state_metrics = {}
        for t in np.linspace(0, 1.1, 50):
            m = ev.evaluate_state(bid, gt, t)
            state_metrics[t] = {'avg_precision': m['precision'], 'avg_recall': m['recall']}
        plot_pr_curve(state_metrics, 'state_30_pr_curve.png', 
                     title=r'State $s^1_{30}$ Precision and Recall vs Recognition Threshold $\theta$')

    # Event log
    print("\nSTEP 11: Event log...")
    display_event_log(dsts, 50)

    # Queries
    print("\nSTEP 12: Location queries (theta=0.5)...")
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
