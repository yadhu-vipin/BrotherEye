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
        probs[zones[0]] = 1.0          # start in zone 1
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

    def apply_transition(self, matched_id, detected_zone, prob):
        entry = self.table.get_occupant(matched_id)
        if not entry:
            return
        old = copy.deepcopy(entry.probs)
        complement = 1.0 - prob
        new_probs = {}
        for z in self.zones:
            if z == detected_zone:
                new_probs[z] = prob + complement * old[z]
            else:
                new_probs[z] = complement * old[z]
        entry.probs = new_probs
        entry.renormalize()


class FaceRecognitionEngine:
    """Eq. 12 – distance-based probability via exponential decay."""

    def __init__(self, db_path):
        self.db = {}
        with open(db_path, 'r') as f:
            raw = json.load(f)
        for occ_id, enc_lists in raw.items():
            self.db[occ_id] = [np.array(e) for e in enc_lists]

    def distance_to_probability(self, distance):
        return max(0.0, min(1.0, math.exp(-DISTANCE_DECAY_FACTOR * distance)))

    def recognize(self, captured_encoding, candidate_ids):
        min_dist   = float('inf')
        matched_id = None
        votes      = []
        for occ in candidate_ids:
            if occ not in self.db:
                continue
            for enc in self.db[occ]:
                dist = face_recognition.face_distance([enc], captured_encoding)[0]
                if face_recognition.compare_faces(
                        [enc], captured_encoding,
                        tolerance=FACE_MATCH_TOLERANCE)[0]:
                    votes.append(occ)
                if dist < min_dist:
                    min_dist   = dist
                    matched_id = occ

        if matched_id and votes.count(matched_id) >= MIN_VOTE_THRESHOLD:
            return matched_id, self.distance_to_probability(min_dist)
        return None, 0.0


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
        self.bsts   = BSTS(self.building_id, self.zones, my_occupants)

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

                matched, prob = self.engine.recognize(
                    enc, self.bsts.registered_occupants)

                if matched:
                    self.bsts.apply_transition(matched, zone, prob)
                    ev = {"timestamp": ts, "building": self.building_id,
                          "zone": zone, "occupant_id": matched,
                          "prob": round(prob, 4)}
                    self.event_history.append(ev)
                    self.save_history()
                    logging.info(
                        f"Recognized {matched} at {zone} (p={prob:.3f})")
                else:
                    logging.info(f"Unknown individual at {zone}")

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
