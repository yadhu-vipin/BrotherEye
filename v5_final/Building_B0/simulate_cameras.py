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

    zones = [f"z1_{bid}", f"z2_{bid}", f"z3_{bid}", f"z4_{bid}", f"zT_{bid}"]

    print(f"Camera simulator for Building {bid}")
    print(f"  Occupants : {', '.join(occupants)}")
    print(f"  Server    : {host}:{port}")
    print(f"  Zones     : {', '.join(zones)}")
    print()

    start_time = datetime.strptime("09:00:00", "%H:%M:%S")
    time_step = timedelta(minutes=24)
    
    # Generate all events first
    all_events = []
    for occ in occupants:
        current_time = start_time
        for i in range(20):
            # Pick a random encoding for this occupant
            if occ in occ_encs:
                base = random.choice(occ_encs[occ])
                noise = np.random.normal(0, 0.01, base.shape)
                captured = base + noise
            else:
                # Should not happen due to check above
                continue

            # Randomly pick a zone for this step
            zone = random.choice(zones)
            ts   = current_time.strftime('%H:%M:%S')

            payload = {
                "type": "LOCAL_EVENT",
                "data": {
                    "zone": zone,
                    "timestamp": ts,
                    "captured_encoding": captured.tolist(),
                    "ground_truth": occ
                }
            }
            all_events.append((current_time, payload))
            current_time += time_step

    # Sort ALL events by timestamp across all occupants
    all_events.sort(key=lambda x: x[0])

    print(f"Starting simulation of {len(all_events)} sorted events...")
    events_sent = 0
    try:
        for dt, payload in all_events:
            ts   = payload["data"]["timestamp"]
            occ  = payload["data"]["ground_truth"]
            zone = payload["data"]["zone"]
            
            if send_event(host, port, payload):
                events_sent += 1
                print(f"  [{ts}] Simulated {occ} at {zone}")
            else:
                print(f"  [{ts}] Server unreachable")
                # We could break here, but maybe it's a transient failure
                time.sleep(1)

            time.sleep(0.05) # Quick burst simulation
        
        print(f"\nSimulation complete. Sent {events_sent} events total.")

    except KeyboardInterrupt:
        print("\nCamera simulator stopped.")

if __name__ == "__main__":
    main()
