import socket
import json
import time
import os
import random
import sys
import numpy as np
from datetime import datetime, timedelta

def send_event(host, port, payload):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3.0)
            s.connect((host, int(port)))
            s.sendall((json.dumps(payload) + "\n").encode('utf-8'))
            return True
    except Exception:
        return False

def main():
    folder   = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(folder, 'config.json')
    db_path  = os.path.join(folder, 'test_db.json')

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    bid        = cfg['my_building_id']
    host       = cfg['buildings'][bid]['host']
    port       = cfg['buildings'][bid]['port']
    occupants  = cfg['buildings'][bid]['occupants']
    zone_conns = cfg.get('zone_connections', {})

    with open(db_path, 'r') as f:
        db = json.load(f)

    # Pre-load encodings for local occupants
    occ_encs = {}
    for occ in occupants:
        if occ in db:
            occ_encs[occ] = [np.array(e) for e in db[occ]]
        else:
            print(f"WARNING: No encodings found for {occ}")

    if not occ_encs:
        print("ERROR: No occupant encodings loaded. Run generate_db first.")
        sys.exit(1)

    # ─── Linear bidirectional zone layout: zT ↔ z1 ↔ z2 ↔ z3 ↔ z4 ──────────
    ZONE_ORDER = [f"zT_{bid}", f"z1_{bid}", f"z2_{bid}", f"z3_{bid}", f"z4_{bid}"]

    def get_adjacent_zones(zone):
        """Return list of zones adjacent to the given zone per the linear graph."""
        neighbors = [z.strip() for z in zone_conns.get(zone, [])]
        return neighbors if neighbors else [zone]

    def next_zone_adjacent(current_zone):
        """Pick a random adjacent zone (realistic movement)."""
        adj = get_adjacent_zones(current_zone)
        return random.choice(adj)

    print(f"Camera simulator for Building {bid}")
    print(f"  Occupants : {', '.join(occupants)}")
    print(f"  Server    : {host}:{port}")
    print(f"  Zones     : {' <-> '.join(ZONE_ORDER)}")
    print(f"  Adjacency : Linear bidirectional graph")
    print()

    start_time = datetime.strptime("09:00:00", "%H:%M:%S")
    time_step = timedelta(minutes=24)
    
    # ─── Generate ADJACENCY-AWARE movement events ────────────────────────────
    # Each occupant starts at zT and moves through adjacent zones only
    all_events = []
    occ_positions = {}  # track current zone per occupant

    for occ in occupants:
        occ_positions[occ] = f"zT_{bid}"  # All start in transition zone
        current_time = start_time

        for i in range(20):
            if occ not in occ_encs:
                continue

            base = random.choice(occ_encs[occ])
            noise = np.random.normal(0, 0.01, base.shape)
            captured = base + noise

            # Move to an adjacent zone (respects adjacency graph)
            new_zone = next_zone_adjacent(occ_positions[occ])
            occ_positions[occ] = new_zone
            ts = current_time.strftime('%H:%M:%S')

            payload = {
                "type": "LOCAL_EVENT",
                "data": {
                    "zone": new_zone,
                    "timestamp": ts,
                    "captured_encoding": captured.tolist(),
                    "ground_truth": occ
                }
            }
            all_events.append((current_time, payload, False))  # False = not teleportation
            current_time += time_step

    # ─── Inject INTENTIONAL TELEPORTATION events ─────────────────────────────
    # These deliberately violate adjacency (e.g., zT → z4, z1 → z4)
    # The server should detect and reject these via State Recovery
    teleport_time = datetime.strptime("14:00:00", "%H:%M:%S")
    teleport_occ = occupants[0]  # First occupant gets teleportation tests

    teleport_pairs = [
        (f"zT_{bid}", f"z3_{bid}"),  # Skip z1, z2
        (f"z1_{bid}", f"z4_{bid}"),  # Skip z2, z3
        (f"z4_{bid}", f"zT_{bid}"),  # Skip z3, z2, z1
    ]

    for idx, (z_from, z_to) in enumerate(teleport_pairs):
        if teleport_occ not in occ_encs:
            continue
        base = random.choice(occ_encs[teleport_occ])
        noise = np.random.normal(0, 0.01, base.shape)
        captured = base + noise
        tp_time = teleport_time + timedelta(minutes=5 * idx)
        ts = tp_time.strftime('%H:%M:%S')

        payload = {
            "type": "LOCAL_EVENT",
            "data": {
                "zone": z_to,
                "timestamp": ts,
                "captured_encoding": captured.tolist(),
                "ground_truth": teleport_occ
            }
        }
        all_events.append((tp_time, payload, True))  # True = teleportation test

    # Sort ALL events by timestamp
    all_events.sort(key=lambda x: x[0])

    total_normal = sum(1 for _, _, tp in all_events if not tp)
    total_teleport = sum(1 for _, _, tp in all_events if tp)
    print(f"Starting simulation: {total_normal} normal + {total_teleport} teleportation events")
    print(f"Teleportation events target: {teleport_occ}")
    print()

    events_sent = 0
    try:
        for dt, payload, is_teleport in all_events:
            ts   = payload["data"]["timestamp"]
            occ  = payload["data"]["ground_truth"]
            zone = payload["data"]["zone"]
            
            tag = " [TELEPORT TEST]" if is_teleport else ""
            
            if send_event(host, port, payload):
                events_sent += 1
                print(f"  [{ts}] {occ} -> {zone}{tag}")
            else:
                print(f"  [{ts}] Server unreachable")
                time.sleep(1)

            time.sleep(0.05)
        
        print(f"\nSimulation complete. Sent {events_sent} events total.")
        print(f"  Normal events:       {total_normal}")
        print(f"  Teleportation tests: {total_teleport}")

    except KeyboardInterrupt:
        print("\nCamera simulator stopped.")

if __name__ == "__main__":
    main()
