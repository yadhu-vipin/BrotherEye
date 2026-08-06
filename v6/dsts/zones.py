"""
v6/dsts/zones.py — Zone topology and adjacency
================================================
Defines the zone graph for each building:
  8 internal zones + 1 transition zone (z_T), single entrance through lobby.

Zone graph (identical per building):
    z4(lab)   z5(office)      z6(lounge)  z7(seminar)
        \\       /                  \\        /
      z2 (corridor-A) ──── z3 (corridor-B) ──── z8 (restricted)
              \\          /
               z1 (lobby)
                   │
                  z_T
"""

import numpy as np
from typing import List, Dict, Tuple, FrozenSet

NUM_INTERNAL_ZONES = 8
NUM_ZONES = 9  # 8 internal + z_T
ZONE_NAMES = [f"z{i}" for i in range(1, NUM_INTERNAL_ZONES + 1)] + ["z_T"]
ZONE_INDEX = {name: i for i, name in enumerate(ZONE_NAMES)}

# Adjacency list — undirected edges in the zone graph
_ADJACENCY_EDGES = [
    ("z_T", "z1"),       # transition zone ↔ lobby
    ("z1", "z2"),        # lobby ↔ corridor-A
    ("z1", "z3"),        # lobby ↔ corridor-B
    ("z2", "z3"),        # corridor-A ↔ corridor-B
    ("z2", "z4"),        # corridor-A ↔ lab
    ("z2", "z5"),        # corridor-A ↔ office
    ("z3", "z6"),        # corridor-B ↔ lounge
    ("z3", "z7"),        # corridor-B ↔ seminar
    ("z3", "z8"),        # corridor-B ↔ restricted
]


def build_adjacency_matrix() -> np.ndarray:
    """Build a boolean adjacency matrix for the zone graph.
    Shape: (NUM_ZONES, NUM_ZONES). Self-loops excluded."""
    adj = np.zeros((NUM_ZONES, NUM_ZONES), dtype=bool)
    for za, zb in _ADJACENCY_EDGES:
        i, j = ZONE_INDEX[za], ZONE_INDEX[zb]
        adj[i, j] = True
        adj[j, i] = True
    return adj


ADJACENCY_MATRIX = build_adjacency_matrix()


def adjacent_zones(zone: str) -> List[str]:
    """Return sorted list of zones adjacent to the given zone."""
    idx = ZONE_INDEX[zone]
    return [ZONE_NAMES[j] for j in range(NUM_ZONES) if ADJACENCY_MATRIX[idx, j]]


def are_adjacent(zone_a: str, zone_b: str) -> bool:
    """Check if two zones are adjacent in the zone graph."""
    return ADJACENCY_MATRIX[ZONE_INDEX[zone_a], ZONE_INDEX[zone_b]]


def zone_label(zone: str) -> str:
    """Human-readable label for a zone."""
    labels = {
        "z1": "lobby", "z2": "corridor-A", "z3": "corridor-B",
        "z4": "lab", "z5": "office", "z6": "lounge",
        "z7": "seminar", "z8": "restricted", "z_T": "transition"
    }
    return labels.get(zone, zone)


def qualified_zone(zone: str, building_id: str) -> str:
    """Return building-qualified zone name, e.g. 'z1_B3'."""
    return f"{zone}_{building_id}"
