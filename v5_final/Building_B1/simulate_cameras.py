import socket
import json
import time
import os
import random
import sys
import numpy as np
from datetime import datetime

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

    try:
        while True:
            # Pick a random occupant and simulate a camera capture
            occ = random.choice(list(occ_encs.keys()))
            base = random.choice(occ_encs[occ])
            noise = np.random.normal(0, 0.01, base.shape)
            captured = base + noise

            zone = random.choice(zones)
            ts   = datetime.now().strftime('%H:%M:%S')

            payload = {
                "type": "LOCAL_EVENT",
                "data": {
                    "zone": zone,
                    "timestamp": ts,
                    "captured_encoding": captured.tolist()
                }
            }

            if send_event(host, port, payload):
                print(f"  [{ts}] Camera captured {occ} at {zone}")
            else:
                print(f"  [{ts}] Server unreachable")

            time.sleep(random.uniform(4.0, 12.0))

    except KeyboardInterrupt:
        print("\nCamera simulator stopped.")

if __name__ == "__main__":
    main()
