import os
import json
import shutil

def generate():
    base_dir = r"d:\brotherEye\BrotherEye\v5_final"
    os.makedirs(base_dir, exist_ok=True)

    buildings = ["B0", "B1", "B2", "B3", "B4"]
    ports = {"B0": 9000, "B1": 9001, "B2": 9002, "B3": 9003, "B4": 9004}

    # =========================================================================
    # building_server.py
    # =========================================================================
    building_server_code = r'''import socket
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
MIN_VOTE_THRESHOLD    = 12
FACE_MATCH_TOLERANCE  = 0.6
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
'''

    # =========================================================================
    # simulate_cameras.py
    # =========================================================================
    simulate_cameras_code = r'''import socket
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
    db_path  = os.path.join(folder, 'encodings_db.json')

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
'''

    # =========================================================================
    # query_network.py
    # =========================================================================
    query_network_code = r'''import socket
import json
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Query the distributed DSTS network for a person's location history.")
    parser.add_argument("--person", required=True,
                        help="Occupant ID to search for (e.g. B1_Person_2)")
    args = parser.parse_args()

    folder   = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(folder, 'config.json')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    bid  = cfg['my_building_id']
    host = cfg['buildings'][bid]['host']
    port = cfg['buildings'][bid]['port']

    print(f"Querying network for '{args.person}' via Building {bid} ({host}:{port})...")
    print()

    req = {"type": "SEARCH_REQ", "occupant_id": args.person, "federated": True}

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(8.0)
            s.connect((host, int(port)))
            s.sendall((json.dumps(req) + "\n").encode('utf-8'))

            buf = b""
            while True:
                chunk = s.recv(16384)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break

            resp = json.loads(buf.decode('utf-8').strip())

    except Exception as e:
        print(f"ERROR: Could not reach building server — {e}")
        print("Make sure building_server.py is running first.")
        sys.exit(1)

    results   = resp.get("results", [])
    snapshots = resp.get("state_snapshots", [])

    # ── Print tracking timeline ──────────────────────────────────────────────
    print("=" * 80)
    print(f"TRACK: {args.person}")
    print("=" * 80)
    print()

    if results:
        print(f"{'Time':<12} {'Building':<10} {'Zone':<16} {'Probability':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r['timestamp']:<12} {r['building']:<10} "
                  f"{r['zone']:<16} {r['prob']:<12.4f}")
        print("-" * 80)
        print(f"Total events: {len(results)}")
    else:
        print("No tracking events found for this person.")

    # ── Print current state table snapshot ────────────────────────────────────
    if snapshots:
        print()
        print("=" * 80)
        print(f"CURRENT STATE TABLE SNAPSHOT")
        print("=" * 80)
        print()
        for snap in snapshots:
            if not snap:
                continue
            print(f"  Building {snap['building']}:")
            for zone, prob in snap['zone_probabilities'].items():
                marker = " *" if prob >= 0.5 else ""
                print(f"    {zone:<20} {prob:.4f}{marker}")
            print(f"    >> Current location: {snap['current_zone']} "
                  f"(p={snap['current_prob']:.4f})")
            print()

    # ── Save to JSON ─────────────────────────────────────────────────────────
    out = os.path.join(folder, f"search_result_{args.person}.json")
    report = {
        "query": args.person,
        "queried_from": bid,
        "tracking_events": results,
        "state_snapshots": snapshots
    }
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to: {out}")

if __name__ == "__main__":
    main()
'''

    # =========================================================================
    # README.txt
    # =========================================================================
    readme_template = '''================================================================================
  DISTRIBUTED STATE TRANSITION SYSTEM (DSTS)
  Building {BID} -- Self-Contained Node
================================================================================

PREREQUISITES
-------------
  Python 3.8+
  pip install face_recognition numpy scikit-learn

SETUP
-----
  1. Open config.json in a text editor.
  2. Replace the IP addresses with everyone's real IPs.
     (If testing locally, leave them as 127.0.0.1)
  3. Make sure encodings_db.json exists in this folder.
     (It should already be included. If not, ask the person
      who set up the system to generate it.)

RUNNING
-------
  Option A -- Double-click run.bat (Windows only)

  Option B -- Two terminal windows:
    Terminal 1:  python building_server.py
    Terminal 2:  python simulate_cameras.py

QUERYING
--------
  Open a third terminal in this folder:

    python query_network.py --person B1_Person_2

  This will search ALL online buildings for that person's
  tracking history and save a detailed JSON report.

OCCUPANT IDS
------------
  Building B0: B0_Person_1 through B0_Person_5
  Building B1: B1_Person_1 through B1_Person_5
  Building B2: B2_Person_1 through B2_Person_5
  Building B3: B3_Person_1 through B3_Person_5
  Building B4: B4_Person_1 through B4_Person_5

FILES
-----
  building_server.py   -- TCP socket server (BSTS engine)
  simulate_cameras.py  -- Simulates 5 occupant camera feeds
  query_network.py     -- Search for any person across buildings
  config.json          -- Network topology (edit IPs here)
  encodings_db.json    -- LFW face encodings for all 25 occupants
  event_history.json   -- Created at runtime, logs all events
  search_result_*.json -- Created when you run a query
================================================================================
'''

    # =========================================================================
    # run.bat
    # =========================================================================
    run_bat_template = r'''@echo off
echo Starting Building {BID} DSTS Node...
echo.
start "Building {BID} Server" cmd /k python building_server.py
ping 127.0.0.1 -n 4 > nul
start "Building {BID} Cameras" cmd /k python simulate_cameras.py
echo.
echo Both processes started in separate windows.
echo To query: python query_network.py --person B1_Person_2
pause
'''

    # =========================================================================
    # generate_db.py (lives at v5 root for initial setup only)
    # =========================================================================
    generate_db_code = r'''import json
import numpy as np
import face_recognition
from sklearn.datasets import fetch_lfw_people
import os
import shutil

IMAGES_PER_PERSON = 40

def augment_encodings(encodings, target_count):
    augmented = list(encodings)
    while len(augmented) < target_count:
        base = encodings[np.random.randint(0, len(encodings))]
        noise = np.random.normal(0, 0.01, base.shape)
        augmented.append(base + noise)
    return augmented[:target_count]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Fetching LFW Dataset...")
    lfw = fetch_lfw_people(min_faces_per_person=15, resize=0.4, color=True)

    all_occupants = []
    for b in ["B0", "B1", "B2", "B3", "B4"]:
        for i in range(1, 6):
            all_occupants.append(f"{b}_Person_{i}")

    encodings_db = {}
    idx = 0
    print("Generating face encodings...")
    for person_name in lfw.target_names:
        if idx >= 25:
            break
        lfw_idx = np.where(lfw.target_names == person_name)[0][0]
        img_idxs = np.where(lfw.target == lfw_idx)[0]

        encs = []
        for ii in img_idxs[:IMAGES_PER_PERSON]:
            img = (lfw.images[ii] * 255).astype(np.uint8)
            found = face_recognition.face_encodings(img)
            if found:
                encs.append(found[0])

        if encs:
            if len(encs) < IMAGES_PER_PERSON:
                encs = augment_encodings(encs, IMAGES_PER_PERSON)
            sid = all_occupants[idx]
            encodings_db[sid] = [e.tolist() for e in encs]
            print(f"  {idx+1:>2}. {sid} ({person_name}) [{len(encs)} encodings]")
            idx += 1

    # Save individual databases per building
    for b in ["B0", "B1", "B2", "B3", "B4"]:
        building_occupants = [f"{b}_Person_{i}" for i in range(1, 6)]
        building_db = {k: v for k, v in encodings_db.items() if k in building_occupants}
        
        dest = os.path.join(base_dir, f"Building_{b}", "encodings_db.json")
        with open(dest, 'w') as f:
            json.dump(building_db, f)
        print(f"  Saved specific DB to Building_{b}/ ({len(building_db)} occupants)")

    print("\nDone! All building folders now have their own specific encodings DB.")

if __name__ == "__main__":
    main()
'''

    # =========================================================================
    # Actually create the folders and files
    # =========================================================================
    for bid in buildings:
        bdir = os.path.join(base_dir, f"Building_{bid}")
        os.makedirs(bdir, exist_ok=True)

        # config.json — unique per building (my_building_id differs)
        cfg = {
            "my_building_id": bid,
            "buildings": {}
        }
        for b in buildings:
            cfg["buildings"][b] = {
                "host": "127.0.0.1",
                "port": ports[b],
                "occupants": [f"{b}_Person_{i}" for i in range(1, 6)]
            }
        with open(os.path.join(bdir, "config.json"), "w", encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)

        # Python scripts
        with open(os.path.join(bdir, "building_server.py"), "w", encoding='utf-8') as f:
            f.write(building_server_code)
        with open(os.path.join(bdir, "simulate_cameras.py"), "w", encoding='utf-8') as f:
            f.write(simulate_cameras_code)
        with open(os.path.join(bdir, "query_network.py"), "w", encoding='utf-8') as f:
            f.write(query_network_code)

        # README
        with open(os.path.join(bdir, "README.txt"), "w", encoding='utf-8') as f:
            f.write(readme_template.replace("{BID}", bid))

        # run.bat
        with open(os.path.join(bdir, "run.bat"), "w", encoding='utf-8') as f:
            f.write(run_bat_template.replace("{BID}", bid))

    # generate_db.py at root
    with open(os.path.join(base_dir, "generate_db.py"), "w", encoding='utf-8') as f:
        f.write(generate_db_code)

    print("V5 generated successfully!")
    print(f"  Location: {base_dir}")
    print()
    for bid in buildings:
        print(f"  Building_{bid}/")
    print()
    print("Next step: Run  python generate_db.py  from the v5_final folder")
    print("           to create the face encodings DB and distribute it.")

if __name__ == "__main__":
    generate()
