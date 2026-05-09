import os
import json

def generate():
    base_dir = r"d:\brotherEye\BrotherEye\v2_sockets"
    os.makedirs(base_dir, exist_ok=True)
    
    # Generate Config
    config = {
        "buildings": {
            "B0": {"host": "127.0.0.1", "port": 6000},
            "B1": {"host": "127.0.0.1", "port": 6001},
            "B2": {"host": "127.0.0.1", "port": 6002},
            "B3": {"host": "127.0.0.1", "port": 6003},
            "B4": {"host": "127.0.0.1", "port": 6004}
        }
    }
    with open(os.path.join(base_dir, "shared_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Building Node Code
    building_code = """import socket
import threading
import json
import time
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(message)s')

class BuildingNode:
    def __init__(self, building_id):
        self.building_id = building_id
        config_path = os.path.join(os.path.dirname(__file__), '..', 'shared_config.json')
        with open(config_path, 'r') as f:
            self.network_config = json.load(f)['buildings']
            
        self.host = self.network_config[building_id]['host']
        self.port = self.network_config[building_id]['port']
        
        self.peers = {b_id: info for b_id, info in self.network_config.items() if b_id != self.building_id}
        self.online_peers = {} # b_id -> last heartbeat time
        
        self.log_file = os.path.join(os.path.dirname(__file__), f"events_{self.building_id}.json")
        self.events = []
        self.load_local_events()
        
        self.timeout_threshold = 10.0
        self.running = True

    def load_local_events(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.events = json.load(f)
            except: pass
            
    def save_local_events(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.events, f, indent=2)

    def send_tcp(self, host, port, payload, timeout=2.0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect((host, port))
                s.sendall((json.dumps(payload) + "\\n").encode('utf-8'))
                
                buffer = ""
                while True:
                    data = s.recv(4096)
                    if not data: break
                    buffer += data.decode('utf-8')
                    if "\\n" in buffer: break
                if buffer.strip():
                    return json.loads(buffer.strip())
        except Exception:
            return None

    def heartbeat_loop(self):
        while self.running:
            hb = {"type": "HEARTBEAT", "building": self.building_id}
            for b_id, info in self.peers.items():
                resp = self.send_tcp(info['host'], info['port'], hb, timeout=1.0)
                if not resp:
                    pass # Offline
            
            # Check timeouts
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
                data = conn.recv(4096)
                if not data: break
                buffer += data.decode('utf-8')
                if "\\n" in buffer: break
            
            if not buffer.strip(): return
            
            try:
                msg = json.loads(buffer.strip())
            except: return
            
            msg_type = msg.get("type")
            
            if msg_type == "LOCAL_EVENT":
                self.events.append(msg["data"])
                self.save_local_events()
                logging.info(f"Local Event: {msg['data']}")
                conn.sendall(b'{"status": "ok"}\\n')
                
            elif msg_type == "HEARTBEAT":
                b_id = msg.get("building")
                if b_id and b_id not in self.online_peers:
                    logging.info(f"{b_id} ONLINE")
                self.online_peers[b_id] = time.time()
                conn.sendall(b'{"status": "ok"}\\n')
                
            elif msg_type == "SEARCH_REQ":
                query = msg.get("query", "").lower()
                is_federated = msg.get("federated", False)
                
                # Local Search
                local_results = [ev for ev in self.events if any(query in str(v).lower() for v in ev.values())]
                all_results = local_results
                
                participating = [self.building_id]
                
                if is_federated:
                    for b_id in list(self.online_peers.keys()):
                        info = self.peers[b_id]
                        req = {"type": "SEARCH_REQ", "query": query, "federated": False}
                        resp = self.send_tcp(info['host'], info['port'], req)
                        if resp and resp.get("type") == "SEARCH_RES":
                            all_results.extend(resp.get("results", []))
                            participating.append(b_id)
                            
                all_results.sort(key=lambda x: x.get('timestamp', ''))
                res = {"type": "SEARCH_RES", "results": all_results, "participating": participating}
                conn.sendall((json.dumps(res) + "\\n").encode('utf-8'))

    def start_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)
        logging.info(f"Node {self.building_id} listening on {self.host}:{self.port}")
        
        hb_thread = threading.Thread(target=self.heartbeat_loop, daemon=True)
        hb_thread.start()
        
        try:
            while self.running:
                conn, addr = server.accept()
                t = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
                t.start()
        except KeyboardInterrupt:
            self.running = False
            server.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--building-id", required=True)
    args = parser.parse_args()
    BuildingNode(args.building_id).start_server()
"""

    # Occupant Laptop Code
    laptop_code = """import socket
import json
import time
import os
import random
import argparse
from datetime import datetime

def send_tcp(host, port, payload):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect((host, port))
            s.sendall((json.dumps(payload) + "\\n").encode('utf-8'))
            buffer = ""
            while True:
                data = s.recv(1024)
                if not data: break
                buffer += data.decode('utf-8')
                if "\\n" in buffer: break
            return True
    except:
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--building-id", required=True)
    parser.add_argument("--occupant-id", required=True)
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'shared_config.json')
    with open(config_path, 'r') as f:
        network_config = json.load(f)['buildings']
        
    host = network_config[args.building_id]['host']
    port = network_config[args.building_id]['port']
    
    print(f"Starting {args.occupant_id} -> {host}:{port}")
    zones = [f"z1_{args.building_id}", f"z2_{args.building_id}", f"z3_{args.building_id}", f"zT_{args.building_id}"]
    
    try:
        while True:
            ev = {
                "building": args.building_id,
                "occupant": args.occupant_id,
                "zone": random.choice(zones),
                "event": random.choice(["ENTRY", "MOVE", "EXIT"]),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            msg = {"type": "LOCAL_EVENT", "data": ev}
            
            if send_tcp(host, port, msg):
                print(f"[{ev['timestamp']}] Sent {ev['event']} at {ev['zone']}")
            else:
                print(f"[{ev['timestamp']}] Failed to send event to Building Node")
                
            time.sleep(random.uniform(5.0, 15.0))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
"""

    # Client Code
    client_code = """import socket
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'shared_config.json')
    with open(config_path, 'r') as f:
        network_config = json.load(f)['buildings']
        
    host = network_config[args.node]['host']
    port = network_config[args.node]['port']
    
    print(f"Connecting to {args.node} at {host}:{port}...")
    
    req = {"type": "SEARCH_REQ", "query": args.query, "federated": True}
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect((host, port))
            s.sendall((json.dumps(req) + "\\n").encode('utf-8'))
            
            buffer = ""
            while True:
                data = s.recv(4096)
                if not data: break
                buffer += data.decode('utf-8')
                if "\\n" in buffer: break
                
            resp = json.loads(buffer.strip())
            
            results = resp.get("results", [])
            print(f"--- Search Results ({len(results)} found) ---")
            for res in results:
                print(f"[{res.get('timestamp')}] {res.get('occupant')} performed {res.get('event')} in {res.get('zone')} (Log from: {res.get('building')})")
            print(f"\\nParticipating Nodes: {', '.join(resp.get('participating', []))}")
            
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    main()
"""

    # Run All Code
    run_all_code = """import subprocess
import time
import os
import sys

def main():
    print("Starting V2 Sockets Distributed System...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processes = []
    
    buildings = ["B0", "B1", "B2", "B3", "B4"]
    for b in buildings:
        b_dir = os.path.join(base_dir, f"Building_{b}")
        script = os.path.join(b_dir, "building_node.py")
        print(f"Starting {b} server...")
        p = subprocess.Popen([sys.executable, script, "--building-id", b], cwd=b_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        
    time.sleep(3)
    
    for b in buildings:
        for i in range(1, 6):
            occ_id = f"{b}_Person_{i}"
            occ_dir = os.path.join(base_dir, f"Laptop_{occ_id}")
            script = os.path.join(occ_dir, "occupant_laptop.py")
            print(f"Starting laptop {occ_id}...")
            p = subprocess.Popen([sys.executable, script, "--building-id", b, "--occupant-id", occ_id], cwd=occ_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(p)
            
    print("\\nAll 30 processes running in background.")
    print("Use Client folder to run searches. Press Ctrl+C to stop.")
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        for p in processes: p.terminate()
        print("Stopped.")

if __name__ == "__main__":
    main()
"""

    # Create Building Folders
    for b in ["B0", "B1", "B2", "B3", "B4"]:
        b_dir = os.path.join(base_dir, f"Building_{b}")
        os.makedirs(b_dir, exist_ok=True)
        with open(os.path.join(b_dir, "building_node.py"), "w") as f:
            f.write(building_code)
            
    # Create Laptop Folders
    for b in ["B0", "B1", "B2", "B3", "B4"]:
        for i in range(1, 6):
            occ_id = f"{b}_Person_{i}"
            occ_dir = os.path.join(base_dir, f"Laptop_{occ_id}")
            os.makedirs(occ_dir, exist_ok=True)
            with open(os.path.join(occ_dir, "occupant_laptop.py"), "w") as f:
                f.write(laptop_code)
                
    # Create Client Folder
    client_dir = os.path.join(base_dir, "Client")
    os.makedirs(client_dir, exist_ok=True)
    with open(os.path.join(client_dir, "client.py"), "w") as f:
        f.write(client_code)
        
    # Create run_all.py
    with open(os.path.join(base_dir, "run_all.py"), "w") as f:
        f.write(run_all_code)
        
    print("V2 System generated successfully!")

if __name__ == "__main__":
    generate()
