import socket
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
                s.sendall((json.dumps(payload) + "\n").encode('utf-8'))
                buffer = ""
                while True:
                    data = s.recv(8192)
                    if not data: break
                    buffer += data.decode('utf-8')
                    if "\n" in buffer: break
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
                if "\n" in buffer: break
                
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
                conn.sendall(b'{"status": "ok"}\n')
                
            elif msg_type == "HEARTBEAT":
                b_id = msg.get("building")
                if b_id and b_id not in self.online_peers: logging.info(f"{b_id} ONLINE")
                self.online_peers[b_id] = time.time()
                conn.sendall(b'{"status": "ok"}\n')
                
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
                conn.sendall((json.dumps({"type": "SEARCH_RES", "results": all_results}) + "\n").encode('utf-8'))

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
