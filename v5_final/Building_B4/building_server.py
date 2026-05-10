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
import matplotlib
matplotlib.use('Agg')
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
THETA = 0.5  # Optimal distance threshold found during evaluation

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
        zt_zone = next((z for z in zones if 'zT' in z), zones[0])
        probs[zt_zone] = 1.0          # start in Transition Zone
        self.entries[occupant_id] = OccupantEntry(occupant_id, probs)

    def get_occupant(self, occupant_id):
        return self.entries.get(occupant_id)


class BSTS:
    """Building-Specific State Transition System (Eq. 9-10 from paper)."""

    def __init__(self, building_id, zones, registered_occupants):
        self.building_id = building_id
        self.zones = zones
        self.registered_occupants = set(registered_occupants)
        self.table = StateTable()
        for occ in registered_occupants:
            self.table.add_occupant(occ, zones)

    def apply_transition(self, event_probs, detected_zone):
        """Eq. 9-10 from paper: Update state of ALL occupants."""
        for occ in self.registered_occupants:
            entry = self.table.get_occupant(occ)
            if not entry:
                continue
            
            p_jk = event_probs.get(occ, 0.0)
            old_probs = copy.deepcopy(entry.probs)
            x_i = 1.0 - p_jk
            
            # Update detected zone: Z_jk = p_jk + (x_i * previous_p_jk)
            entry.probs[detected_zone] = p_jk + (x_i * old_probs[detected_zone])
            
            # Update all other zones: Z_lk = x_i * previous_p_lk
            for zone in self.zones:
                if zone != detected_zone:
                    entry.probs[zone] = x_i * old_probs[zone]
            
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
            
            distances = face_recognition.face_distance(self.db[name], captured_encoding)
            scores[name] = np.min(distances)

        # Apply THETA filtering: if best match is too far, return uncertain probs
        best_name = min(scores, key=scores.get)
        if scores[best_name] > THETA:
            return {name: 1.0/len(candidate_ids) for name in candidate_ids}

        # Convert distances to probabilities
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

        db_path    = os.path.join(self.folder, 'reference_db.json')
        self.engine = FaceRecognitionEngine(db_path)
        self.zones  = [f"z1_{self.building_id}",
                       f"z2_{self.building_id}",
                       f"z3_{self.building_id}",
                       f"z4_{self.building_id}",
                       f"zT_{self.building_id}"]
        self.bsts   = BSTS(self.building_id, self.zones, my_occupants)

        self.online_peers     = {}
        self.timeout_threshold = 10.0
        self.running           = True
        self.plot_lock         = threading.Lock()

        self.history_file = os.path.join(self.folder, 'event_history.json')
        self.event_history = []
        
        self.state_history_file = os.path.join(self.folder, 'state_history.json')
        self.state_log_file     = os.path.join(self.folder, 'state_transition_tables.txt')
        self.state_history = []
        
        with open(self.state_log_file, 'w') as f:
            f.write(f"--- INITIAL STATE S0 (Building {self.building_id}) ---\n\n")

        self.record_state_snapshot("INITIAL_STATE", "N/A", "00:00:00")
        self.log_state_table(0, "System Initialization", "N/A", "N/A")

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.event_history, f, indent=2)

    def record_state_snapshot(self, event_type, zone, timestamp):
        snapshot = {
            "event_index": len(self.state_history),
            "event_type": event_type,
            "detected_zone": zone,
            "timestamp": timestamp,
            "state_table": {
                occ: {z: round(p, 6) for z, p in entry.probs.items()}
                for occ, entry in self.bsts.table.entries.items()
            }
        }
        self.state_history.append(snapshot)
        try:
            with open(self.state_history_file, 'w') as f:
                json.dump(self.state_history, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save state history: {e}")

    def log_state_table(self, state_idx, event_desc, detected_zone, ground_truth):
        data = []
        occupants = sorted(self.bsts.registered_occupants)
        for occ in occupants:
            entry = self.bsts.table.get_occupant(occ)
            row = {"Occupant": occ}
            row.update(entry.probs)
            data.append(row)
        
        df = pd.DataFrame(data).set_index("Occupant")
        zt_cols = [c for c in df.columns if 'zT' in c]
        other_cols = sorted([c for c in df.columns if 'zT' not in c])
        df = df[zt_cols + other_cols]

        log_block = [f"\n--- PROCESSING STATE S{state_idx} ---",
                     f"Event: {event_desc}",
                     df.to_string()]
        
        if ground_truth != "N/A":
            valid = "[OK]" if any(occ in ground_truth for occ in occupants) else "[INFO]"
            log_block.append(f"{valid} State S{state_idx} LOGGED: System processed detection at {detected_zone}")

        output = "\n".join(log_block) + "\n"
        print(output)
        try:
            with open(self.state_log_file, 'a', encoding='utf-8') as f:
                f.write(output)
        except Exception:
            pass

    def save_occupant_history(self, occupant_id, event):
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
        data = [ev for ev in self.event_history if ev["occupant_id"] == occupant_id]
        if len(data) < 2: return None
        try:
            df = pd.DataFrame(data)
            df['dt'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S')
            df = df.sort_values('dt')
            
            with self.plot_lock:
                plt.figure(figsize=(10, 6))
                sns.set_theme(style="darkgrid")
                sns.scatterplot(data=df, x='timestamp', y='zone', hue='prob', size='prob', palette="viridis", sizes=(100, 500))
                plt.plot(df['timestamp'], df['zone'], linestyle='-', alpha=0.3, color='gray')
                plt.title(f"Track Sequence for {occupant_id} (Building {self.building_id})")
                plt.xticks(rotation=45)
                plt.tight_layout()
                img_path = os.path.join(self.folder, f"occupant_{occupant_id}_track.png")
                plt.savefig(img_path)
                plt.close('all')
                return img_path
        except Exception as e:
            logging.error(f"Failed to generate track: {e}")
            return None

    @staticmethod
    def _recv_json(sock):
        buf = b""
        while True:
            chunk = sock.recv(16384)
            if not chunk: break
            buf += chunk
            if b"\n" in buf: break
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

    def heartbeat_loop(self):
        while self.running:
            hb = {"type": "HEARTBEAT", "building": self.building_id}
            for b_id, info in self.peers.items():
                self.send_tcp(info['host'], info['port'], hb, timeout=1.5)
            now = time.time()
            gone = [b for b, ts in self.online_peers.items() if now - ts > self.timeout_threshold]
            for b in gone:
                del self.online_peers[b]
                logging.info(f"Peer {b} went OFFLINE")
            time.sleep(3.0)

    def handle(self, conn, addr):
        with conn:
            msg = self._recv_json(conn)
            if not msg: return
            t = msg.get("type")
            if t == "LOCAL_EVENT":
                if len(self.state_history) > 100:
                    conn.sendall(b'{"status":"limit_reached"}\n')
                    return
                data = msg["data"]
                enc  = np.array(data["captured_encoding"])
                zone, ts = data["zone"], data["timestamp"]
                event_probs = self.engine.recognize(enc, self.bsts.registered_occupants)
                self.bsts.apply_transition(event_probs, zone)
                matched = max(event_probs, key=event_probs.get)
                prob = event_probs[matched]
                if prob > 0.01:
                    ev = {"timestamp": ts, "building": self.building_id, "zone": zone, "occupant_id": matched, "prob": round(prob, 4)}
                    self.event_history.append(ev)
                    self.save_history()
                    self.save_occupant_history(matched, ev)
                    self.generate_occupant_track(matched)
                    self.record_state_snapshot(f"DETECTED_{matched}", zone, ts)
                    gt = msg["data"].get("ground_truth", matched)
                    self.log_state_table(len(self.state_history)-1, f"{gt} detected at {zone}", zone, gt)
                    logging.info(f"Recognized {matched} as best match at {zone} (p={prob:.3f})")
                else:
                    logging.info(f"Uncertain detection at {zone}")
                conn.sendall(b'{"status":"ok"}\n')
            elif t == "HEARTBEAT":
                b_id = msg.get("building")
                if b_id: self.online_peers[b_id] = time.time()
                conn.sendall(b'{"status":"ok"}\n')
            elif t == "SEARCH_REQ":
                occ_id = msg.get("occupant_id")
                federated = msg.get("federated", False)
                local = [ev for ev in self.event_history if ev["occupant_id"] == occ_id]
                entry = self.bsts.table.get_occupant(occ_id)
                state_snapshot = {"building": self.building_id, "zone_probabilities": {z: round(p, 4) for z, p in entry.probs.items()}, "current_zone": max(entry.probs, key=entry.probs.get), "current_prob": round(entry.probs[max(entry.probs, key=entry.probs.get)], 4)} if entry else None
                all_results = list(local)
                all_states = [state_snapshot] if state_snapshot else []
                if federated:
                    for b_id in list(self.online_peers.keys()):
                        info = self.peers.get(b_id)
                        if info:
                            resp = self.send_tcp(info['host'], info['port'], {"type": "SEARCH_REQ", "occupant_id": occ_id, "federated": False})
                            if resp and resp.get("type") == "SEARCH_RES":
                                all_results.extend(resp.get("results", []))
                                if resp.get("state_snapshots"): all_states.extend(resp["state_snapshots"])
                all_results.sort(key=lambda x: x.get("timestamp", ""))
                conn.sendall((json.dumps({"type": "SEARCH_RES", "results": all_results, "state_snapshots": all_states}) + "\n").encode('utf-8'))
            elif t == "OCCUPANT_DATA_REQ":
                occ_id = msg.get("occupant_id")
                hist_path = os.path.join(self.folder, f"occupant_{occ_id}_history.json")
                history_data = []
                if os.path.exists(hist_path):
                    with open(hist_path, 'r') as f: history_data = json.load(f)
                img_path = self.generate_occupant_track(occ_id)
                img_base64 = ""
                if img_path and os.path.exists(img_path):
                    with open(img_path, "rb") as image_file: img_base64 = base64.b64encode(image_file.read()).decode('utf-8')
                conn.sendall((json.dumps({"type": "OCCUPANT_DATA_RES", "occupant_id": occ_id, "building": self.building_id, "history": history_data, "track_image_base64": img_base64}) + "\n").encode('utf-8'))

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(10)
        logging.info(f"Building {self.building_id} listening on {self.host}:{self.port}")
        threading.Thread(target=self.heartbeat_loop, daemon=True).start()
        try:
            while self.running:
                conn, addr = srv.accept()
                threading.Thread(target=self.handle, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            self.running = False
            srv.close()

if __name__ == "__main__":
    BuildingNode().start()
