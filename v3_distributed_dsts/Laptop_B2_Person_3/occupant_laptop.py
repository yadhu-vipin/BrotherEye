import socket
import json
import time
import os
import random
import argparse
import numpy as np
from datetime import datetime

def send_tcp(host, port, payload):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((host, port))
            s.sendall((json.dumps(payload) + "\n").encode('utf-8'))
            return True
    except: return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--building-id", required=True)
    parser.add_argument("--occupant-id", required=True)
    args = parser.parse_args()
    
    # Load Network Config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'shared_config.json')
    with open(config_path, 'r') as f:
        network_config = json.load(f)['buildings']
    host, port = network_config[args.building_id]['host'], network_config[args.building_id]['port']
    
    # Load Encodings DB
    db_path = os.path.join(os.path.dirname(__file__), '..', 'shared_encodings_db.json')
    with open(db_path, 'r') as f:
        db = json.load(f)
        
    my_encodings = [np.array(e) for e in db.get(args.occupant_id, [])]
    if not my_encodings:
        print(f"Error: No face data found for {args.occupant_id} in DB!")
        return

    print(f"Starting Camera Simulator for {args.occupant_id} -> {host}:{port}")
    zones = [f"z1_{args.building_id}", f"z2_{args.building_id}", f"z3_{args.building_id}", f"z4_{args.building_id}", f"zT_{args.building_id}"]
    
    try:
        while True:
            # Pick a random encoding and add noise
            base = random.choice(my_encodings)
            noise = np.random.normal(0, 0.01, base.shape)
            captured = base + noise
            
            zone = random.choice(zones)
            ev = {
                "zone": zone,
                "captured_encoding": captured.tolist()
            }
            msg = {"type": "LOCAL_EVENT", "data": ev}
            
            if send_tcp(host, port, msg):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Streamed face capture from {zone}")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Connection failed")
                
            time.sleep(random.uniform(5.0, 15.0))
    except KeyboardInterrupt: pass

if __name__ == "__main__":
    main()
