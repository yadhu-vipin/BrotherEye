import pytest
from datetime import datetime, timedelta

from dsts_simulation import ZoneGraph, DSTS, BUILDING_ZONE_CONFIGS, BSTS


def test_zone_graph_shortest_path():
    zg = ZoneGraph(BUILDING_ZONE_CONFIGS, intra_zone_time=10.0, inter_building_time=120.0)
    # pick two sequential zones within building 0
    tz = 'z1_b0'
    nz = 'z3_b0'
    tt = zg.shortest_travel_time(tz, nz)
    assert tt is not None
    assert tt >= 0


def test_infer_possible_locations_simple():
    # create DSTS with two small buildings and two occupants
    dsts = DSTS()
    # build small BSTS per BUILDING_ZONE_CONFIGS using default occupants
    for bid, zones in BUILDING_ZONE_CONFIGS.items():
        occs = [f"B{bid}_P1"]
        bsts = BSTS(bid, zones, occs)
        dsts.register_building(bsts)

    # create a fake event for the occupant in building 0 at zone z1_b0
    occ_id = 'B0_P1'
    now = dsts.current_time
    # create a RecognitionEvent-like object by submitting a recognition with mocked encoding
    # We cannot call recognition in tests; instead, directly set the occupant probs to a delta at z1_b0
    b0 = dsts.buildings[0]
    entry = b0.get_occupant_entry(occ_id)
    assert entry is not None
    # set a delta distribution at z1_b0
    for z in entry.probs:
        entry.probs[z] = 0.0
    entry.probs['z1_b0'] = 1.0

    # add a synthetic last event to the global log
    from dsts_simulation import RecognitionEvent
    ev = RecognitionEvent(timestamp=now - timedelta(seconds=30), building_id=0,
                          zone_id='z1_b0', event_counter=1,
                          probabilities={'B0_P1': 0.98}, matched_occupant=occ_id)
    dsts.global_event_log.append(ev)

    inferred = dsts.infer_possible_locations(occ_id, at_time=now)
    assert isinstance(inferred, dict)
    # z1_b0 should be reachable and present
    assert 'z1_b0' in inferred
    # normalized
    s = sum(inferred.values())
    assert abs(s - 1.0) < 1e-6