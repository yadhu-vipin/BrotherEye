import os
import json

def generate():
    base_dir = r"d:\brotherEye\BrotherEye\v4_distributed_laptops"
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate Config
    config = {
        "buildings": {
            "B0": {"host": "127.0.0.1", "port": 9000, "occupants": ["B0_Person_1", "B0_Person_2", "B0_Person_3", "B0_Person_4", "B0_Person_5"]},
            "B1": {"host": "127.0.0.1", "port": 9001, "occupants": ["B1_Person_1", "B1_Person_2", "B1_Person_3", "B1_Person_4", "B1_Person_5"]},
            "B2": {"host": "127.0.0.1", "port": 9002, "occupants": ["B2_Person_1", "B2_Person_2", "B2_Person_3", "B2_Person_4", "B2_Person_5"]},
            "B3": {"host": "127.0.0.1", "port": 9003, "occupants": ["B3_Person_1", "B3_Person_2", "B3_Person_3", "B3_Person_4", "B3_Person_5"]},
            "B4": {"host": "127.0.0.1", "port": 9004, "occupants": ["B4_Person_1", "B4_Person_2", "B4_Person_3", "B4_Person_4", "B4_Person_5"]}
        }
    }
    with open(os.path.join(base_dir, "shared_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 1. generate_db.py
    generate_db_code = """import json
import numpy as np
import face_recognition
from sklearn.datasets import fetch_lfw_people
import os

IMAGES_PER_PERSON = 40

def augment_encodings(encodings, target_count):
    augmented = list(encodings)
    while len(augmented) < target_count:
        base = encodings[np.random.randint(0, len(encodings))]
        noise = np.random.normal(0, 0.01, base.shape)
        augmented.append(base + noise)
    return augmented[:target_count]

def main():
    print("Fetching LFW Dataset... This will take a few minutes.")
    lfw_data = fetch_lfw_people(min_faces_per_person=15, resize=0.4, color=True)
    
    encodings_db = {}
    person_idx = 0
    all_occupants = []
    for b in ["B0", "B1", "B2", "B3", "B4"]:
        for i in range(1, 6):
            all_occupants.append(f"{b}_Person_{i}")
            
    print("Generating Encodings...")
    for person_name in lfw_data.target_names:
        if person_idx >= 25: break
        idx_in_lfw = np.where(lfw_data.target_names == person_name)[0][0]
        image_indices = np.where(lfw_data.target == idx_in_lfw)[0]
        enc_list = []
        for idx in image_indices[:IMAGES_PER_PERSON]:
            img_uint8 = (lfw_data.images[idx] * 255).astype(np.uint8)
            encs = face_recognition.face_encodings(img_uint8)
            if encs: enc_list.append(encs[0])
                
        if len(enc_list) > 0:
            if len(enc_list) < IMAGES_PER_PERSON:
                enc_list = augment_encodings(enc_list, IMAGES_PER_PERSON)
            sys_id = all_occupants[person_idx]
            encodings_db[sys_id] = [e.tolist() for e in enc_list]
            print(f"Generated {IMAGES_PER_PERSON} encodings for {sys_id}")
            person_idx += 1

    db_path = os.path.join(os.path.dirname(__file__), 'shared_encodings_db.json')
    with open(db_path, 'w') as f:
        json.dump(encodings_db, f)
    print(f"Saved DB to {db_path}")

if __name__ == "__main__":
    main()
"""

    # 2. building_node.py
    building_code = """import socket
import threading
import json
import time
import os
import logging
import argparse
import math
import numpy as np
import copy
import face_recognition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(message)s')

DISTANCE_DECAY_FACTOR = 5.0
MIN_VOTE_THRESHOLD = 12
FACE_MATCH_TOLERANCE = 0.6

class OccupantEntry:
    def __init__(self, occupant_id, probs):
        self.occupant_id = occupant_id
        self.probs = probs
    def renormalize(self):
        total = sum(self.probs.values())
        if total > 0:
            self.probs = {z: p/total for z, p in self.probs.items()}

class StateTable:
    def __init__(self):
        self.entries = {}
    def add_occupant(self, occupant_id, zones):
        prob_per_zone = 1.0 / len(zones)
        probs = {zone: prob_per_zone for zone in zones}
        probs[zones[0]] = 1.0
        for z in zones[1:]: probs[z] = 0.0
        self.entries[occupant_id] = OccupantEntry(occupant_id, probs)
    def get_occupant(self, occupant_id):
        return self.entries.get(occupant_id)
        
class BSTS:
    def __init__(self, building_id, zones, registered_occupants):
        self.building_id = building_id
        self.zones = zones
        self.registered_occupants = set(registered_occupants)
        self.table = StateTable()
        for occ in registered_occupants:
            self.table.add_occupant(occ, zones)
            
    def apply_transition(self, matched_id, detected_zone, prob):
        entry = self.table.get_occupant(matched_id)
        if not entry: return
        current_probs = copy.deepcopy(entry.probs)
        x_i = 1.0 - prob
        new_probs = {}
        for z in self.zones:
            if z == detected_zone:
                new_probs[z] = prob + x_i * current_probs[z]
            else:
                new_probs[z] = x_i * current_probs[z]
        entry.probs = new_probs
        entry.renormalize()

class FaceRecognitionEngine:
    def __init__(self, db_path):
        self.db = {}
        with open(db_path, 'r') as f:
            raw_db = json.load(f)
            for occ, enc_lists in raw_db.items():
                self.db[occ] = [np.array(e) for e in enc_lists]
                
    def distance_to_probability(self, distance):
        p = math.exp(-DISTANCE_DECAY_FACTOR * distance)
        return max(0.0, min(1.0, p))

    def recognize(self, captured_encoding, candidate_ids):
        min_distance = float('inf')
        matched_id = None
        votes = []
        for occ in candidate_ids:
            if occ not in self.db: continue
            for enc in self.db[occ]:
                dist = face_recognition.face_distance([enc], captured_encoding)[0]
                is_match = face_recognition.compare_faces([enc], captured_encoding, tolerance=FACE_MATCH_TOLERANCE)[0]
                if is_match: votes.append(occ)
                if dist < min_distance:
                    min_distance = dist
                    matched_id = occ
        vote_count = votes.count(matched_id) if matched_id else 0
        if matched_id and vote_count >= MIN_VOTE_THRESHOLD:
            return matched_id, self.distance_to_probability(min_distance)
        return None, 0.0

class BuildingNode:
    def __init__(self, building_id):
        self.building_id = building_id
        config_path = os.path.join(os.path.dirname(__file__), '..', 'shared_config.json')
        with open(config_path, 'r') as f:
            self.network_config = json.load(f)['buildings']
            
        self.host = self.network_config[building_id]['host']
        self.port = self.network_config[building_id]['port']
        self.peers = {b: info for b, info in self.network_config.items() if b != self.building_id}
        self.online_peers = {}
        
        db_path = os.path.join(os.path.dirname(__file__), '..', 'shared_encodings_db.json')
        self.engine = FaceRecognitionEngine(db_path)
        self.zones = [f"z1_{building_id}", f"z2_{building_id}", f"z3_{building_id}", f"z4_{building_id}", f"zT_{building_id}"]
        self.bsts = BSTS(building_id, self.zones, self.network_config[building_id]['occupants'])
        
        self.timeout_threshold = 10.0
        self.running = True
        
        # Load local history
        self.history_file = os.path.join(os.path.dirname(__file__), f"event_history_{building_id}.json")
        self.event_history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.event_history = json.load(f)
            except: pass

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.event_history, f, indent=2)

    def send_tcp(self, host, port, payload, timeout=2.0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((host, port))
                s.sendall((json.dumps(payload) + "\\n").encode('utf-8'))
                buffer = ""
                while True:
                    data = s.recv(8192)
                    if not data: break
                    buffer += data.decode('utf-8')
                    if "\\n" in buffer: break
                if buffer.strip():
                    return json.loads(buffer.strip())
        except: return None

    def heartbeat_loop(self):
        while self.running:
            hb = {"type": "HEARTBEAT", "building": self.building_id}
            for b_id, info in self.peers.items():
                self.send_tcp(info['host'], info['port'], hb, timeout=1.0)
                
            curr_time = time.time()
            offline = [b for b, ts in self.online_peers.items() if curr_time - ts > self.timeout_threshold]
            for b in offline:
                del self.online_peers[b]
                logging.info(f"{b} OFFLINE")
            time.sleep(3.0)

    def handle_client(self, conn, addr):
        with conn:
            buffer = ""
            while True:
                data = conn.recv(16384)
                if not data: break
                buffer += data.decode('utf-8')
                if "\\n" in buffer: break
                
            if not buffer.strip(): return
            try: msg = json.loads(buffer.strip())
            except: return
            
            msg_type = msg.get("type")
            
            if msg_type == "LOCAL_EVENT":
                data = msg["data"]
                captured_encoding = np.array(data["captured_encoding"])
                zone = data["zone"]
                timestamp = data["timestamp"]
                
                matched_id, prob = self.engine.recognize(captured_encoding, self.bsts.registered_occupants)
                if matched_id:
                    self.bsts.apply_transition(matched_id, zone, prob)
                    logging.info(f"DSTS: Recognized {matched_id} at {zone} (p={prob:.3f})")
                    
                    self.event_history.append({
                        "timestamp": timestamp,
                        "building": self.building_id,
                        "zone": zone,
                        "occupant_id": matched_id,
                        "prob": prob
                    })
                    self.save_history()
                conn.sendall(b'{"status": "ok"}\\n')
                
            elif msg_type == "HEARTBEAT":
                b_id = msg.get("building")
                if b_id and b_id not in self.online_peers: logging.info(f"{b_id} ONLINE")
                self.online_peers[b_id] = time.time()
                conn.sendall(b'{"status": "ok"}\\n')
                
            elif msg_type == "SEARCH_REQ":
                occ_id = msg.get("occupant_id")
                is_federated = msg.get("federated", False)
                
                # Local Query
                all_results = [ev for ev in self.event_history if ev["occupant_id"] == occ_id]
                
                if is_federated:
                    for b_id in list(self.online_peers.keys()):
                        info = self.peers[b_id]
                        req = {"type": "SEARCH_REQ", "occupant_id": occ_id, "federated": False}
                        resp = self.send_tcp(info['host'], info['port'], req)
                        if resp and resp.get("type") == "SEARCH_RES":
                            all_results.extend(resp.get("results", []))
                            
                # Sort combined results by timestamp
                all_results.sort(key=lambda x: x["timestamp"])
                conn.sendall((json.dumps({"type": "SEARCH_RES", "results": all_results}) + "\\n").encode('utf-8'))

    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)
        logging.info(f"Node {self.building_id} listening on {self.host}:{self.port}")
        
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        try:
            while self.running:
                conn, addr = server.accept()
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            self.running = False
            server.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--building-id", required=True)
    args = parser.parse_args()
    BuildingNode(args.building_id).start_server()
"""

    # 3. occupant_laptop.py (DSTS Face Capture & Search Client)
    laptop_code = """import socket
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
            s.sendall((json.dumps(payload) + "\\n").encode('utf-8'))
            buffer = ""
            while True:
                data = s.recv(16384)
                if not data: break
                buffer += data.decode('utf-8')
                if "\\n" in buffer: break
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
            
        print(f"\\n{'=' * 80}")
        print(f"TRACK: {args.query}")
        print(f"{'=' * 80}\\n")
        
        if not results:
            print(f"No tracking data found for {args.query}")
        else:
            print(f"{'Time':<15} {'Bldg':<10} {'Zone':<12} {'Prob':<10}")
            print("-" * 80)
            for r in results:
                print(f"{r['timestamp']:<15} B{r['building']:<9} {r['zone']:<12} {r['prob']:<10.3f}")
            print("-" * 80)
            print(f"Total: {len(results)}")
            print(f"\\nDetailed JSON report saved to {out_file}\\n")
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
"""

    # 4. run_all.py
    run_all_code = """import subprocess
import time
import os
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, 'shared_encodings_db.json')
    
    if not os.path.exists(db_path):
        print("Initial setup: Generating LFW Face Encodings DB...")
        subprocess.run([sys.executable, os.path.join(base_dir, 'generate_db.py')])
        
    print("\\nStarting V4 DSTS Socket Architecture...")
    processes = []
    buildings = ["B0", "B1", "B2", "B3", "B4"]
    
    for b in buildings:
        b_dir = os.path.join(base_dir, f"Building_{b}")
        print(f"Starting {b} BSTS Node...")
        p = subprocess.Popen([sys.executable, "building_node.py", "--building-id", b], cwd=b_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        
    time.sleep(3)
    
    for b in buildings:
        for i in range(1, 6):
            occ_id = f"{b}_Person_{i}"
            occ_dir = os.path.join(base_dir, f"Laptop_{occ_id}")
            print(f"Starting camera laptop {occ_id}...")
            p = subprocess.Popen([sys.executable, "occupant_laptop.py", "--building-id", b, "--occupant-id", occ_id], cwd=occ_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(p)
            
    print("\\nAll 30 processes running in background. DSTS logic is active.")
    print("\\nTo search, use an occupant laptop terminal:\\n")
    print("  cd Laptop_B0_Person_5")
    print("  python occupant_laptop.py --building-id B0 --occupant-id B0_Person_5 --query B1_Person_2\\n")
    print("Press Ctrl+C to stop.")
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        for p in processes: p.terminate()
        print("Stopped.")

if __name__ == "__main__":
    main()
"""

    # Write files
    with open(os.path.join(base_dir, "generate_db.py"), "w") as f: f.write(generate_db_code)
    with open(os.path.join(base_dir, "run_all.py"), "w") as f: f.write(run_all_code)
    
    for b in ["B0", "B1", "B2", "B3", "B4"]:
        b_dir = os.path.join(base_dir, f"Building_{b}")
        os.makedirs(b_dir, exist_ok=True)
        with open(os.path.join(b_dir, "building_node.py"), "w") as f: f.write(building_code)
            
        for i in range(1, 6):
            occ_id = f"{b}_Person_{i}"
            occ_dir = os.path.join(base_dir, f"Laptop_{occ_id}")
            os.makedirs(occ_dir, exist_ok=True)
            with open(os.path.join(occ_dir, "occupant_laptop.py"), "w") as f: f.write(laptop_code)
                
    print("V4 System generated successfully!")

if __name__ == "__main__":
    generate()
