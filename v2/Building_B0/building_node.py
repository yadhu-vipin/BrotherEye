import socket
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
                s.sendall((json.dumps(payload) + "\n").encode('utf-8'))
                
                buffer = ""
                while True:
                    data = s.recv(4096)
                    if not data: break
                    buffer += data.decode('utf-8')
                    if "\n" in buffer: break
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
                if "\n" in buffer: break
            
            if not buffer.strip(): return
            
            try:
                msg = json.loads(buffer.strip())
            except: return
            
            msg_type = msg.get("type")
            
            if msg_type == "LOCAL_EVENT":
                self.events.append(msg["data"])
                self.save_local_events()
                logging.info(f"Local Event: {msg['data']}")
                conn.sendall(b'{"status": "ok"}\n')
                
            elif msg_type == "HEARTBEAT":
                b_id = msg.get("building")
                if b_id and b_id not in self.online_peers:
                    logging.info(f"{b_id} ONLINE")
                self.online_peers[b_id] = time.time()
                conn.sendall(b'{"status": "ok"}\n')
                
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
                conn.sendall((json.dumps(res) + "\n").encode('utf-8'))

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
