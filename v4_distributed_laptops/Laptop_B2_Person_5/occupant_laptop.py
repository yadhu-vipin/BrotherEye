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
            s.settimeout(5.0)
            s.connect((host, port))
            s.sendall((json.dumps(payload) + "\n").encode('utf-8'))
            buffer = ""
            while True:
                data = s.recv(16384)
                if not data: break
                buffer += data.decode('utf-8')
                if "\n" in buffer: break
            if buffer.strip():
                return json.loads(buffer.strip())
            return True
    except: return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--building-id", required=True)
    parser.add_argument("--occupant-id", required=True)
    parser.add_argument("--query", help="Target occupant ID to search for (triggers search mode)")
    args = parser.parse_args()
    
    # Load Network Config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'shared_config.json')
    with open(config_path, 'r') as f:
        network_config = json.load(f)['buildings']
    host, port = network_config[args.building_id]['host'], network_config[args.building_id]['port']
    
    # ------------------ SEARCH MODE ------------------
    if args.query:
        print(f"[{args.occupant_id}] Querying network for {args.query} via {args.building_id}...")
        req = {"type": "SEARCH_REQ", "occupant_id": args.query, "federated": True}
        resp = send_tcp(host, port, req)
        
        if not resp:
            print("Failed to reach building node.")
            return
            
        results = resp.get("results", [])
        
        # Save to local JSON
        out_file = os.path.join(os.path.dirname(__file__), f"search_result_{args.query}.json")
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\n{'=' * 80}")
        print(f"TRACK: {args.query}")
        print(f"{'=' * 80}\n")
        
        if not results:
            print(f"No tracking data found for {args.query}")
        else:
            print(f"{'Time':<15} {'Bldg':<10} {'Zone':<12} {'Prob':<10}")
            print("-" * 80)
            for r in results:
                print(f"{r['timestamp']:<15} B{r['building']:<9} {r['zone']:<12} {r['prob']:<10.3f}")
            print("-" * 80)
            print(f"Total: {len(results)}")
            print(f"\nDetailed JSON report saved to {out_file}\n")
        return

    # ------------------ GENERATION MODE ------------------
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
    
    local_log_file = os.path.join(os.path.dirname(__file__), "my_events.json")
    my_events = []
    if os.path.exists(local_log_file):
        try:
            with open(local_log_file, 'r') as f:
                my_events = json.load(f)
        except: pass

    try:
        while True:
            base = random.choice(my_encodings)
            noise = np.random.normal(0, 0.01, base.shape)
            captured = base + noise
            
            zone = random.choice(zones)
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            ev = {
                "zone": zone,
                "timestamp": timestamp,
                "captured_encoding": captured.tolist()
            }
            msg = {"type": "LOCAL_EVENT", "data": ev}
            
            if send_tcp(host, port, msg):
                print(f"[{timestamp}] Streamed face capture from {zone}")
                
                # Log locally
                my_events.append({"timestamp": timestamp, "zone": zone, "action": "STREAMED_ENCODING"})
                with open(local_log_file, 'w') as f:
                    json.dump(my_events, f, indent=2)
            else:
                print(f"[{timestamp}] Connection failed")
                
            time.sleep(random.uniform(5.0, 15.0))
    except KeyboardInterrupt: pass

if __name__ == "__main__":
    main()
