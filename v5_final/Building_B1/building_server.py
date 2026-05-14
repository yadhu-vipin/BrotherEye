import socket
import threading
import json
import time
import os
import logging
import math
import numpy as np
import copy
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64

try:
    import face_recognition
except ImportError:
    print("ERROR: 'face_recognition' package not installed.")
    print("Install it with: pip install face_recognition")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# ─── Constants ───────────────────────────────────────────────────────────────
DISTANCE_DECAY_FACTOR = 5.0
MIN_VOTE_THRESHOLD    = 5
FACE_MATCH_TOLERANCE  = 0.7
THETA                 = 0.5
PROB_FLOOR            = 0.001

# ─── DSTS Math ───────────────────────────────────────────────────────────────

class OccupantEntry:
    def __init__(self, occupant_id, probs):
        self.occupant_id = occupant_id
        self.probs = probs

    def renormalize(self):
        total = sum(self.probs.values())
        if total > 0:
            self.probs = {z: p / total for z, p in self.probs.items()}


class StateTable:
    def __init__(self):
        self.entries = {}

    def add_occupant(self, occupant_id, zones):
        probs = {z: 0.0 for z in zones}
        # Find the transition zone (contains 'zT')
        zt_zone = next((z for z in zones if 'zT' in z), zones[0])
        probs[zt_zone] = 1.0          # start in Transition Zone
        self.entries[occupant_id] = OccupantEntry(occupant_id, probs)

    def get_occupant(self, occupant_id):
        return self.entries.get(occupant_id)


class BSTS:
    """Building-Specific State Transition System (Eq. 9-10 from paper)."""

    def __init__(self, building_id, zones, registered_occupants, zone_connections=None):
        self.building_id = building_id
        self.zones = zones
        self.registered_occupants = set(registered_occupants)
        self.zone_connections = zone_connections if zone_connections is not None else {}
        self.table = StateTable()
        for occ in registered_occupants:
            self.table.add_occupant(occ, zones)

    def reason_movement(self, raw_event_probs, detected_zone):
        """Spatio-Temporal Reasoning and Maximum Difference Criterion."""
        d_zone = detected_zone.strip()
        weighted_probs = {}
        
        # 1. Spatio-Temporal Reasoning: scale probabilities via Adjacency Matrix
        for occ in self.registered_occupants:
            p_raw = raw_event_probs.get(occ, 0.0)
            entry = self.table.get_occupant(occ)
            if not entry or not entry.probs:
                weighted_probs[occ] = p_raw
                continue
            
            # Find current most probable zone z_a
            z_a = max(entry.probs, key=entry.probs.get).strip()
            
            # Check adjacency
            neighbors = [z.strip() for z in self.zone_connections.get(z_a, [])]
            if d_zone in neighbors or z_a == d_zone:
                scale = 1.0
            else:
                scale = 0.1
                
            weighted_probs[occ] = p_raw * scale
            
        # Normalize weighted probabilities
        total_weight = sum(weighted_probs.values())
        if total_weight > 0:
            weighted_probs = {occ: p / total_weight for occ, p in weighted_probs.items()}
        else:
            weighted_probs = {occ: 1.0 / len(self.registered_occupants) for occ in self.registered_occupants}
            
        # 2. Maximum Difference Criterion: argmax_i |p_{s_k}(o_i) - p_{s_{k-1}}(o_i)|
        diff_scores = {}
        for occ in self.registered_occupants:
            entry = self.table.get_occupant(occ)
            old_p = entry.probs.get(d_zone, 0.0) if entry else 0.0
            diff_scores[occ] = weighted_probs.get(occ, 0.0) * (1.0 - old_p)
            
        moved_occupant = max(diff_scores, key=diff_scores.get) if diff_scores else max(weighted_probs, key=weighted_probs.get)
        
        return weighted_probs, moved_occupant, diff_scores

    def apply_transition(self, event_probs, detected_zone):
        """Eq. 9-10 from paper: Update state of ALL occupants."""
        for occ in self.registered_occupants:
            entry = self.table.get_occupant(occ)
            if not entry:
                continue
            
            p_jk = event_probs.get(occ, 0.0)
            old_probs = copy.deepcopy(entry.probs)
            x_i = 1.0 - p_jk
            
            d_zone = detected_zone.strip()
            if d_zone not in entry.probs:
                continue

            # Update detected zone: Z_jk = p_jk + (x_i * previous_p_jk)
            entry.probs[d_zone] = p_jk + (x_i * old_probs[d_zone])
            
            # Update all other zones: Z_lk = x_i * previous_p_lk
            for zone in self.zones:
                z = zone.strip()
                if z != d_zone:
                    entry.probs[z] = max(x_i * old_probs[z], PROB_FLOOR)
            
            entry.renormalize()


class FaceRecognitionEngine:
    """Eq. 12 – distance-based probability via exponential decay."""

    def __init__(self, db_path):
        self.db = {}
        with open(db_path, 'r') as f:
            raw = json.load(f)
        for occ_id, enc_lists in raw.items():
            self.db[occ_id] = [np.array(e) for e in enc_lists]

    def recognize(self, captured_encoding, candidate_ids, lambda_scale=15.0):
        """Eq. 12 - softmax-like probability via exponential decay."""
        scores = {}
        for name in candidate_ids:
            if name not in self.db:
                scores[name] = 1.0 # Max distance if no references exist
                continue
            
            # Calculate Euclidean distances and use the minimum
            distances = face_recognition.face_distance(self.db[name], captured_encoding)
            scores[name] = np.min(distances)

        # Convert distances to probabilities using Softmax + Lambda Scaling
        exponents = {name: math.exp(-lambda_scale * dist) for name, dist in scores.items()}
        sum_exponents = sum(exponents.values())

        probs = {name: exp_val / (sum_exponents + 1e-9) for name, exp_val in exponents.items()}
        return probs


# ─── Building Node ───────────────────────────────────────────────────────────

class BuildingNode:

    def __init__(self):
        self.folder = os.path.dirname(os.path.abspath(__file__))
        cfg_path   = os.path.join(self.folder, 'config.json')
        with open(cfg_path, 'r') as f:
            cfg = json.load(f)

        self.building_id = cfg['my_building_id']
        self.host        = cfg['buildings'][self.building_id]['host']
        self.port        = cfg['buildings'][self.building_id]['port']
        my_occupants     = cfg['buildings'][self.building_id]['occupants']
        self.peers       = {b: info for b, info in cfg['buildings'].items()
                            if b != self.building_id}

        # DSTS components
        db_path    = os.path.join(self.folder, 'encodings_db.json')
        self.engine = FaceRecognitionEngine(db_path)
        self.zones  = [f"z1_{self.building_id}",
                       f"z2_{self.building_id}",
                       f"z3_{self.building_id}",
                       f"z4_{self.building_id}",
                       f"zT_{self.building_id}"]
        self.zone_connections = cfg.get('zone_connections', {})
        self.bsts   = BSTS(self.building_id, self.zones, my_occupants, self.zone_connections)

        # Peer status
        self.online_peers     = {}
        self.timeout_threshold = 10.0
        self.running           = True

        # Event history (persisted in building folder)
        self.history_file = os.path.join(self.folder, 'event_history.json')
        self.event_history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.event_history = json.load(f)
            except Exception:
                pass

    # ── persistence ──────────────────────────────────────────────────────────

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.event_history, f, indent=2)

    def save_occupant_history(self, occupant_id, event):
        """Saves a separate JSON for each individual."""
        file_path = os.path.join(self.folder, f"occupant_{occupant_id}_history.json")
        history = []
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    history = json.load(f)
            except Exception:
                pass
        history.append(event)
        with open(file_path, 'w') as f:
            json.dump(history, f, indent=2)

    def generate_occupant_track(self, occupant_id):
        """Generates a sequence diagram/timeline using Seaborn."""
        data = [ev for ev in self.event_history if ev["occupant_id"] == occupant_id]
        if len(data) < 2:
            return None # Not enough data for a track plot
        
        try:
            df = pd.DataFrame(data)
            # Ensure chronological order
            df['dt'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S')
            df = df.sort_values('dt')
            
            plt.figure(figsize=(10, 6))
            sns.set_theme(style="darkgrid")
            
            # Simple timeline plot
            plot = sns.scatterplot(data=df, x='timestamp', y='zone', 
                                   hue='prob', size='prob', 
                                   palette="viridis", sizes=(100, 500))
            
            # Add lines connecting the points to show the 'track'
            plt.plot(df['timestamp'], df['zone'], linestyle='-', alpha=0.3, color='gray')
            
            plt.title(f"Track Sequence for {occupant_id} (Building {self.building_id})")
            plt.xlabel("Time")
            plt.ylabel("Zone")
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            img_path = os.path.join(self.folder, f"occupant_{occupant_id}_track.png")
            plt.savefig(img_path)
            plt.close()
            return img_path
        except Exception as e:
            logging.error(f"Failed to generate track for {occupant_id}: {e}")
            return None

    # ── TCP helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _recv_json(sock):
        buf = b""
        while True:
            chunk = sock.recv(16384)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break
        text = buf.decode('utf-8').strip()
        return json.loads(text) if text else None

    def send_tcp(self, host, port, payload, timeout=3.0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((host, int(port)))
                s.sendall((json.dumps(payload) + "\n").encode('utf-8'))
                return self._recv_json(s)
        except Exception:
            return None

    # ── heartbeat ────────────────────────────────────────────────────────────

    def heartbeat_loop(self):
        while self.running:
            hb = {"type": "HEARTBEAT", "building": self.building_id}
            for b_id, info in self.peers.items():
                self.send_tcp(info['host'], info['port'], hb, timeout=1.5)

            now = time.time()
            gone = [b for b, ts in self.online_peers.items()
                    if now - ts > self.timeout_threshold]
            for b in gone:
                del self.online_peers[b]
                logging.info(f"Peer {b} went OFFLINE")
            time.sleep(3.0)

    # ── connection handler ───────────────────────────────────────────────────

    def handle(self, conn, addr):
        with conn:
            msg = self._recv_json(conn)
            if not msg:
                return

            t = msg.get("type")

            # ── camera event ─────────────────────────────────────────────────
            if t == "LOCAL_EVENT":
                data = msg["data"]
                enc  = np.array(data["captured_encoding"])
                zone = data["zone"]
                ts   = data["timestamp"]

                # Recognition returns raw probabilities for ALL occupants
                raw_event_probs = self.engine.recognize(enc, self.bsts.registered_occupants)
                
                # Spatio-temporal reasoning & Maximum Difference Criterion
                event_probs, matched, diff_scores = self.bsts.reason_movement(raw_event_probs, zone)
                
                logging.info(f"Maximum Difference scores at {zone.strip()}: " + 
                             ", ".join([f"{k}: {v:.4f}" for k, v in diff_scores.items()]))
                
                # Apply transition to ALL occupants using the spatio-temporally weighted probabilities
                self.bsts.apply_transition(event_probs, zone)

                # Use the moved occupant as the matched winner
                prob = event_probs[matched]

                if prob > 0.01: # Threshold for logging
                    ev = {"timestamp": ts, "building": self.building_id,
                          "zone": zone, "occupant_id": matched,
                          "prob": round(prob, 4)}
                    self.event_history.append(ev)
                    self.save_history()
                    
                    # NEW: Per-occupant tracking
                    self.save_occupant_history(matched, ev)
                    self.generate_occupant_track(matched)

                    logging.info(
                        f"Recognized {matched} as best match at {zone} (p={prob:.3f})")
                else:
                    logging.info(f"Uncertain detection at {zone}")

                conn.sendall(b'{"status":"ok"}\n')

            # ── heartbeat ────────────────────────────────────────────────────
            elif t == "HEARTBEAT":
                b_id = msg.get("building")
                if b_id and b_id not in self.online_peers:
                    logging.info(f"Peer {b_id} is ONLINE")
                if b_id:
                    self.online_peers[b_id] = time.time()
                conn.sendall(b'{"status":"ok"}\n')

            # ── search ───────────────────────────────────────────────────────
            elif t == "SEARCH_REQ":
                occ_id     = msg.get("occupant_id")
                federated  = msg.get("federated", False)

                # local results
                local = [ev for ev in self.event_history
                         if ev["occupant_id"] == occ_id]

                # current state table snapshot
                entry = self.bsts.table.get_occupant(occ_id)
                state_snapshot = None
                if entry:
                    max_zone = max(entry.probs, key=entry.probs.get)
                    state_snapshot = {
                        "building": self.building_id,
                        "zone_probabilities": {z: round(p, 4)
                                               for z, p in entry.probs.items()},
                        "current_zone": max_zone,
                        "current_prob": round(entry.probs[max_zone], 4)
                    }

                all_results = list(local)
                all_states  = [state_snapshot] if state_snapshot else []

                if federated:
                    for b_id in list(self.online_peers.keys()):
                        info = self.peers.get(b_id)
                        if not info:
                            continue
                        req = {"type": "SEARCH_REQ",
                               "occupant_id": occ_id, "federated": False}
                        resp = self.send_tcp(info['host'], info['port'], req)
                        if resp and resp.get("type") == "SEARCH_RES":
                            all_results.extend(resp.get("results", []))
                            if resp.get("state_snapshots"):
                                all_states.extend(resp["state_snapshots"])

                all_results.sort(key=lambda x: x.get("timestamp", ""))

                reply = {"type": "SEARCH_RES",
                         "results": all_results,
                         "state_snapshots": all_states}
                conn.sendall((json.dumps(reply) + "\n").encode('utf-8'))

            # ── occupant data retrieval ──────────────────────────────────────
            elif t == "OCCUPANT_DATA_REQ":
                occ_id = msg.get("occupant_id")
                
                # Load JSON history
                hist_path = os.path.join(self.folder, f"occupant_{occ_id}_history.json")
                history_data = []
                if os.path.exists(hist_path):
                    with open(hist_path, 'r') as f:
                        history_data = json.load(f)
                
                # Get/Generate Track Image
                img_path = self.generate_occupant_track(occ_id)
                img_base64 = ""
                if img_path and os.path.exists(img_path):
                    with open(img_path, "rb") as image_file:
                        img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                
                reply = {
                    "type": "OCCUPANT_DATA_RES",
                    "occupant_id": occ_id,
                    "building": self.building_id,
                    "history": history_data,
                    "track_image_base64": img_base64
                }
                conn.sendall((json.dumps(reply) + "\n").encode('utf-8'))

            # ── connection verification ──────────────────────────────────────
            elif t == "CHECK_CONNECTION":
                z1 = msg.get("zone1", "").strip()
                z2 = msg.get("zone2", "").strip()
                neighbors = [z.strip() for z in self.zone_connections.get(z1, [])]
                connected = (z2 in neighbors) or (z1 == z2)
                reply = {"type": "CONNECTION_RES", "connected": connected}
                conn.sendall((json.dumps(reply) + "\n").encode('utf-8'))

    # ── main loop ────────────────────────────────────────────────────────────

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(10)
        logging.info(
            f"Building {self.building_id} listening on "
            f"{self.host}:{self.port}")

        threading.Thread(target=self.heartbeat_loop, daemon=True).start()

        try:
            while self.running:
                conn, addr = srv.accept()
                threading.Thread(target=self.handle,
                                 args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            logging.info("Shutting down.")
            self.running = False
            srv.close()


if __name__ == "__main__":
    BuildingNode().start()
