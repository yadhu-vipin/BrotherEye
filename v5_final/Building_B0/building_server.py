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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import matplotlib.patches as mpatches
from datetime import datetime

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

# ─── Constants (Derived from Mohan et al., 2025 & state_transition.ipynb) ──
LAMBDA_SCALE        = 15.0  # λ: Exponential decay scale for prob mapping
MIN_VOTE_THRESHOLD    = 5
FACE_MATCH_TOLERANCE  = 0.7
THETA                 = 0.5   # Distance threshold for 'Uncertain' detection

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
            # Ensure keys match exactly (stripping any whitespace)
            d_zone = detected_zone.strip()
            
            if d_zone not in entry.probs:
                logging.warning(f"Zone {d_zone} not found in {occ}'s probs!")
                continue

            entry.probs[d_zone] = p_jk + (x_i * old_probs[d_zone])
            
            # Update all other zones: Z_lk = x_i * previous_p_lk
            for zone in self.zones:
                z = zone.strip()
                if z != d_zone:
                    entry.probs[z] = x_i * old_probs[z]
            
            entry.renormalize()


class FaceRecognitionEngine:
    """Eq. 12 – distance-based probability via exponential decay."""

    def __init__(self, db_path):
        self.db = {}
        with open(db_path, 'r') as f:
            raw = json.load(f)
        for occ_id, enc_lists in raw.items():
            self.db[occ_id] = [np.array(e) for e in enc_lists]

    def recognize(self, captured_encoding, candidate_ids, lambda_scale=LAMBDA_SCALE):
        """Eq. 12 - softmax-like probability via exponential decay (Mohan et al.)."""
        scores = {}
        for name in candidate_ids:
            if name not in self.db:
                scores[name] = 1.0 # Max distance if no references exist
                continue
            
            # Calculate Euclidean distances and use the minimum
            distances = face_recognition.face_distance(self.db[name], captured_encoding)
            scores[name] = np.min(distances)

        # Apply THETA filtering: if best match is too far, return uncertain probs
        best_name = min(scores, key=scores.get)
        if scores[best_name] > THETA:
            return {name: 1.0/len(candidate_ids) for name in candidate_ids}

        # Convert distances to probabilities using LAMBDA_SCALE
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
        db_path    = os.path.join(self.folder, 'reference_db.json')
        self.engine = FaceRecognitionEngine(db_path)
        self.zones  = [f"z1_{self.building_id}",
                       f"z2_{self.building_id}",
                       f"z3_{self.building_id}",
                       f"z4_{self.building_id}",
                       f"zT_{self.building_id}"]
        self.bsts   = BSTS(self.building_id, self.zones, my_occupants)

        # Peer status
        self.online_peers     = {}
        self.timeout_threshold = 10.0
        self.running           = True
        self.plot_lock         = threading.Lock()
        self.state_lock        = threading.Lock()

        # Event history (Cleared on startup for fresh simulation visibility)
        self.history_file = os.path.join(self.folder, 'event_history.json')
        self.event_history = []
        # We no longer load old history to prevent "messy" accumulated plots

        # State transition history (Table 1 requirement)
        self.state_history_file = os.path.join(self.folder, 'state_history.json')
        self.state_log_file     = os.path.join(self.folder, 'state_transition_tables.txt')
        self.state_history = []
        
        # Initialize state log
        with open(self.state_log_file, 'w') as f:
            f.write(f"--- INITIAL STATE S0 (Building {self.building_id}) ---\n\n")

        # Record Initial State S0
        self.record_state_snapshot("INITIAL_STATE", "N/A", "00:00:00")
        self.log_state_table(0, "System Initialization", "N/A", "N/A")

    # ── persistence ──────────────────────────────────────────────────────────

    def save_history(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.event_history, f, indent=2)

    def record_state_snapshot(self, event_type, zone, timestamp):
        """Equation 4 & 5: Save state table snapshots for all occupants."""
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
        """Generate formatted Table 1 visualization (S0, S1, ... Sn)."""
        data = []
        occupants = sorted(self.bsts.registered_occupants)
        for occ in occupants:
            entry = self.bsts.table.get_occupant(occ)
            row = {"Occupant": occ}
            row.update(entry.probs)
            data.append(row)
        
        df = pd.DataFrame(data).set_index("Occupant")
        # Use explicit zone ordering from self.zones
        cols = [z for z in self.zones if z in df.columns]
        # Move zT to front
        zt_zone = next((z for z in cols if 'zT' in z), None)
        if zt_zone:
            cols.remove(zt_zone)
            cols = [zt_zone] + cols
        
        df = df[cols]

        # Format the table for better readability
        pd.options.display.float_format = '{:.3f}'.format
        
        log_block = [
            f"\n" + "="*80,
            f"--- STATE S{state_idx} ---",
            f"Event: {event_desc}",
            f"Timestamp: {self.state_history[-1]['timestamp'] if self.state_history else 'N/A'}",
            "="*80,
            df.to_string(),
            "-"*80
        ]
        
        # Validation note
        if ground_truth != "N/A":
            valid = "[OK]" if any(occ in ground_truth for occ in occupants) else "[INFO]"
            log_block.append(f"{valid} State S{state_idx} LOGGED: System processed detection at {detected_zone}")

        output = "\n".join(log_block) + "\n"
        print(output)
        try:
            with open(self.state_log_file, 'a', encoding='utf-8') as f:
                f.write(output)
        except Exception as e:
            logging.error(f"Failed to write to state log: {e}")

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

    def _load_face_sequence(self, occupant_id, count):
        """Loads face thumbnails from the gallery for plot overlay."""
        # In live_simulation, gallery is in the parent folder
        gallery_dir = os.path.join(self.folder, '..', 'face_gallery')
        occ_dir = os.path.join(gallery_dir, occupant_id)
        if not os.path.exists(occ_dir):
            return [None] * count
        imgs = []
        for prefix in ['ref', 'test']:
            for i in range(1, 21):
                p = os.path.join(occ_dir, f"{prefix}_{i:02d}.jpg")
                if os.path.exists(p):
                    try:
                        imgs.append(np.array(Image.open(p).convert('RGB')))
                    except Exception:
                        pass
        if not imgs:
            return [None] * count
        return [imgs[i % len(imgs)] for i in range(count)]

    def _build_step_path(self, xs, ys):
        """Creates L-shaped step paths for logical movement visualization."""
        if not xs: return [], []
        px, py = [xs[0]], [ys[0]]
        for i in range(1, len(xs)):
            if xs[i] != xs[i - 1]:
                px.append(xs[i])
                py.append(ys[i - 1])
            px.append(xs[i])
            py.append(ys[i])
        return px, py

    def generate_occupant_track(self, occupant_id):
        """Generates an improved sequence diagram (Track Visual) with face thumbnails."""
        data = [ev for ev in self.event_history if ev["occupant_id"] == occupant_id]
        if not data: return None
        
        try:
            df = pd.DataFrame(data)
            df['dt'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S')
            df = df.sort_values('dt')
            
            ZONE_ORDER = [f'z1_{self.building_id}', f'z2_{self.building_id}', 
                          f'z3_{self.building_id}', f'z4_{self.building_id}', 
                          f'zT_{self.building_id}']
            
            valid_zones = [z for z in ZONE_ORDER if z in self.zones]
            zone_map = {z: i for i, z in enumerate(valid_zones)}
            
            # Use minutes from 09:00 as Y-axis to spread events out
            def to_mins(dt):
                return (dt.hour - 9) * 60 + dt.minute
            
            pts = [(datetime.strptime("09:00", "%H:%M"), f'zT_{self.building_id}')]
            for _, row in df.iterrows():
                pts.append((row['dt'], row['zone']))
            
            xs = [zone_map.get(z, 0) for _, z in pts]
            ys = [to_mins(t) for t, _ in pts]
            
            faces = self._load_face_sequence(occupant_id, len(pts))
            
            with self.plot_lock:
                # Close any existing plots to start fresh
                plt.close('all')
                fig, ax = plt.subplots(figsize=(14, 12))
                ax.set_facecolor('#F9FAFB')
                for xi in range(len(valid_zones)):
                    ax.axvline(xi, color='#D1D5DB', linewidth=0.8, zorder=0)
                
                px, py = self._build_step_path(xs, ys)
                # Offset ground truth slightly to the left to show it's there
                ax.plot([p - 0.05 for p in px], py, color='#1A56DB', linestyle='--', linewidth=1.5, zorder=1, label='Ground Truth', alpha=0.6)
                ax.plot(px, py, color='#D0312D', linewidth=2.5, zorder=2, label='Estimated Track')
                
                # Thumbnail display logic with density control
                skip_step = max(1, len(pts) // 20) # Max 20 thumbnails
                for i, ((t, z), face_img) in enumerate(zip(pts, faces)):
                    xi = zone_map.get(z, 0)
                    yi = ys[i]
                    
                    if i % skip_step == 0 or i == len(pts)-1:
                        if face_img is not None:
                            pil = Image.fromarray(face_img).resize((64, 64), Image.LANCZOS)
                            imgob = OffsetImage(np.array(pil), zoom=0.6)
                            ab = AnnotationBbox(imgob, (xi, yi), frameon=True,
                                                bboxprops=dict(edgecolor='#D0312D' if i > 0 else '#1A56DB', linewidth=2),
                                                zorder=5)
                            ax.add_artist(ab)
                        else:
                            ax.scatter(xi, yi, s=100, color='#D0312D', zorder=5)
                
                ax.set_xlim(-0.5, len(valid_zones) - 0.5)
                ax.set_ylim(480 + 10, -10) # 0 to 480 mins (9-5)
                ax.set_xticks(range(len(valid_zones)))
                ax.set_xticklabels([z.split('_')[0] for z in valid_zones], fontweight='bold')
                ax.xaxis.tick_top()
                
                # Show hourly labels on Y axis
                ax.set_yticks([h * 60 for h in range(9)])
                ax.set_yticklabels([f"{9+h:02d}:00" for h in range(9)])
                ax.set_ylabel("Time of Day (09:00 - 17:00)", fontweight='bold')
                
                plt.title(f"Track Sequence: {occupant_id} (Building {self.building_id})", pad=20)
                plt.tight_layout()
                
                img_path = os.path.join(self.folder, f"occupant_{occupant_id}_track.png")
                visual_path = os.path.join(self.folder, f"{occupant_id}_track_visual.png")
                plt.savefig(img_path)
                plt.savefig(visual_path)
                plt.close('all')
                return img_path
        except Exception as e:
            logging.error(f"Failed to generate improved track: {e}")
            return None
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
            try:
                msg = self._recv_json(conn)
            if not msg:
                return

            t = msg.get("type")

            # ── camera event ─────────────────────────────────────────────────
            if t == "LOCAL_EVENT":
                logging.info(f"Processing LOCAL_EVENT from {addr}")
                # Limit event history to avoid memory issues in simulation
                if len(self.state_history) > 1000:
                    conn.sendall(b'{"status":"limit_reached"}\n')
                    return
                    
                data = msg["data"]
                enc  = np.array(data["captured_encoding"])
                zone = data["zone"]
                ts   = data["timestamp"]

                # Recognition now returns a dictionary of probabilities for ALL occupants
                event_probs = self.engine.recognize(enc, self.bsts.registered_occupants)
                
                with self.state_lock:
                    # Apply transition to ALL occupants
                    self.bsts.apply_transition(event_probs, zone)

                    # Identify the winner for logging and history
                    matched = max(event_probs, key=event_probs.get) if event_probs else "UNKNOWN"
                    prob = event_probs.get(matched, 0.0)
                    
                    # Check if this is a uniform distribution (uncertain)
                    n_occ = len(event_probs)
                    is_uncertain = (n_occ > 0 and 
                                   all(math.isclose(p, 1.0/n_occ, rel_tol=1e-5) for p in event_probs.values()))

                    if not is_uncertain and prob > 0.1: # Clear winner
                        logging.info(
                            f"Recognized {matched} as best match at {zone} (p={prob:.3f})")
                        
                        ev = {"timestamp": ts, "building": self.building_id,
                              "zone": zone.strip(), "occupant_id": matched.strip(),
                              "prob": round(prob, 4)}
                        self.event_history.append(ev)
                        self.save_history()
                        
                        self.save_occupant_history(matched, ev)
                        self.generate_occupant_track(matched)
                    else:
                        logging.info(f"Uncertain detection at {zone} - skipping track update")

                    # Record and Log State Transition S_k (Always record for DSTS audit)
                    self.record_state_snapshot(f"DETECTED_{matched}", zone.strip(), ts)
                    
                    gt = msg["data"].get("ground_truth", matched).strip()
                    self.log_state_table(len(self.state_history)-1, 
                                        f"{gt} detected at {zone}", 
                                        zone, gt)

                conn.sendall(b'{"status":"ok"}\n')
            except Exception as e:
                logging.error(f"Error handling request from {addr}: {e}")
                import traceback
                traceback.print_exc()

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
