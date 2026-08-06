"""
v6/sim/mobility.py — CTMC mobility on zone adjacency graph
=============================================================
Continuous-time Markov chain on the zone adjacency graph with
log-normal dwell times per zone type.

v5 used random.choice(zones) — occupants teleported, making
track-based reasoning meaningless. v6 enforces adjacency.

Roaming: at z1 (lobby), exit to z_T with r_exit=0.08.
From z_T, return home with 0.55, else pick destination by
gravity model (prob ∝ 1/distance²).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from ..dsts.zones import (
    ZONE_NAMES, ZONE_INDEX, NUM_ZONES, adjacent_zones, ADJACENCY_MATRIX
)
from .campus import gravity_probability

# Zone-type dwell times (log-normal parameters)
DWELL_PARAMS = {
    "z1":  (3.0, 0.5),   # lobby: ~20 min median
    "z2":  (3.5, 0.4),   # corridor-A: ~33 min
    "z3":  (3.5, 0.4),   # corridor-B: ~33 min
    "z4":  (4.5, 0.6),   # lab: ~90 min
    "z5":  (4.2, 0.5),   # office: ~67 min
    "z6":  (3.8, 0.5),   # lounge: ~45 min
    "z7":  (4.0, 0.4),   # seminar: ~55 min
    "z8":  (4.0, 0.6),   # restricted: ~55 min
    "z_T": (2.0, 0.8),   # transition: ~7 min
}

R_EXIT = 0.08       # probability of exiting to z_T from z1
R_RETURN_HOME = 0.55  # probability of returning home from z_T


class MobilityModel:
    """
    CTMC mobility on the zone adjacency graph.

    Generates realistic movement traces where every consecutive
    zone pair is adjacent (or same). Adjacency violations are
    injected deliberately via spurious_rate for testing the
    reasoning module.
    """

    def __init__(
        self,
        building_id: str,
        rng: np.random.Generator,
        spurious_rate: float = 0.05,
    ):
        self.building_id = building_id
        self.rng = rng
        self.spurious_rate = spurious_rate

    def sample_dwell_time(self, zone: str) -> float:
        """Sample a dwell time (minutes) from log-normal distribution."""
        mu, sigma = DWELL_PARAMS.get(zone, (3.5, 0.5))
        return float(self.rng.lognormal(mu, sigma)) / 60.0  # convert to minutes

    def next_zone(
        self,
        current_zone: str,
        home_building: str,
    ) -> Tuple[str, Optional[str]]:
        """
        Sample next zone from the adjacency graph.

        Returns:
            (next_zone, destination_building_or_None)
            If destination_building is not None, occupant is roaming.
        """
        # Deliberate spurious injection
        if self.rng.random() < self.spurious_rate:
            # Pick any non-adjacent zone (guaranteed violation)
            adj = set(adjacent_zones(current_zone)) | {current_zone}
            non_adj = [z for z in ZONE_NAMES if z not in adj]
            if non_adj:
                return self.rng.choice(non_adj), None

        if current_zone == "z1":
            # From lobby: small chance of exiting
            if self.rng.random() < R_EXIT:
                return "z_T", None

        if current_zone == "z_T":
            # From transition zone: return home or roam
            if self.rng.random() < R_RETURN_HOME:
                return "z1", None  # re-enter through lobby
            else:
                # Pick destination building by gravity model
                probs = gravity_probability(home_building)
                buildings = list(probs.keys())
                weights = [probs[b] for b in buildings]
                dest = self.rng.choice(buildings, p=weights)
                return "z_T", dest  # arrive at dest's z_T

        # Normal movement: pick uniformly from adjacent zones
        adj = adjacent_zones(current_zone)
        return self.rng.choice(adj), None

    def generate_trace(
        self,
        occupant_id: str,
        home_building: str,
        start_time: float,
        end_time: float,
    ) -> List[Dict]:
        """
        Generate a movement trace for one occupant across a day.

        Args:
            occupant_id: occupant identifier
            home_building: home building ID
            start_time: simulation start (minutes from midnight)
            end_time: simulation end (minutes from midnight)

        Returns:
            List of event dicts with sim_time, building, zone, occupant
        """
        events = []
        current_zone = "z_T"
        current_building = home_building
        t = start_time

        # Initial entry through lobby
        t += self.sample_dwell_time("z_T")
        current_zone = "z1"
        events.append({
            "sim_time": t,
            "building_id": current_building,
            "zone": current_zone,
            "occupant_id": occupant_id,
        })

        while t < end_time:
            dwell = self.sample_dwell_time(current_zone)
            t += dwell

            if t >= end_time:
                break

            next_z, dest_building = self.next_zone(
                current_zone, current_building
            )

            if dest_building is not None:
                # Inter-building roaming
                current_building = dest_building
                current_zone = "z_T"
            else:
                current_zone = next_z

            events.append({
                "sim_time": t,
                "building_id": current_building,
                "zone": current_zone,
                "occupant_id": occupant_id,
            })

        return events
