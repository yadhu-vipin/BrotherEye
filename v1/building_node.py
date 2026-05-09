import argparse
import json
import logging
import os
import threading
import time
from typing import Dict, List
import requests
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(message)s')

app = Flask(__name__)

class BuildingNode:
    def __init__(self, building_id: str):
        self.building_id = building_id
        
        # Load network config
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        with open(config_path, 'r') as f:
            self.network_config = json.load(f)['buildings']
            
        self.host_url = self.network_config[self.building_id]
        port_str = self.host_url.split(':')[-1]
        self.port = int(port_str)

        # Peer tracking
        self.peers = {b_id: url for b_id, url in self.network_config.items() if b_id != self.building_id}
        self.online_peers: Dict[str, float] = {}  # building_id -> last heartbeat timestamp
        
        # Storage
        self.log_file = os.path.join(os.path.dirname(__file__), f"events_{self.building_id}.json")
        self.events: List[dict] = []
        self.load_local_events()

        # Threading for heartbeats
        self.heartbeat_interval = 3.0
        self.timeout_threshold = 10.0
        self.running = True

    def load_local_events(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.events = json.load(f)
            except json.JSONDecodeError:
                self.events = []
    
    def save_local_events(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.events, f, indent=2)

    def add_event(self, event: dict):
        self.events.append(event)
        self.save_local_events()
        
    def local_search(self, query: str) -> List[dict]:
        query = query.lower()
        results = []
        for ev in self.events:
            # Search over all string values in the event JSON
            if any(query in str(v).lower() for v in ev.values()):
                results.append(ev)
        return results

    def heartbeat_loop(self):
        while self.running:
            self._send_heartbeats()
            self._check_timeouts()
            time.sleep(self.heartbeat_interval)

    def _send_heartbeats(self):
        heartbeat_data = {
            "building": self.building_id,
            "status": "ONLINE",
            "timestamp": time.time()
        }
        for b_id, url in self.peers.items():
            try:
                requests.post(f"{url}/heartbeat", json=heartbeat_data, timeout=1.0)
            except requests.exceptions.RequestException:
                pass # Node is probably offline

    def _check_timeouts(self):
        current_time = time.time()
        offline_nodes = []
        for b_id, last_seen in list(self.online_peers.items()):
            if current_time - last_seen > self.timeout_threshold:
                offline_nodes.append(b_id)
        
        for b_id in offline_nodes:
            del self.online_peers[b_id]
            logging.info(f"{b_id} OFFLINE (Heartbeat timeout)")

    def receive_heartbeat(self, b_id: str):
        if b_id not in self.online_peers:
            logging.info(f"{b_id} ONLINE")
        self.online_peers[b_id] = time.time()

    def federated_search(self, query: str) -> List[dict]:
        # Local search first
        all_results = self.local_search(query)
        
        # Distributed search across online peers
        current_online_peers = list(self.online_peers.keys())
        logging.info(f"Broadcasting search to online peers: {current_online_peers}")
        
        for b_id in current_online_peers:
            url = self.peers[b_id]
            try:
                resp = requests.get(f"{url}/search", params={"q": query}, timeout=3.0)
                if resp.status_code == 200:
                    peer_results = resp.json().get('results', [])
                    all_results.extend(peer_results)
            except requests.exceptions.RequestException:
                logging.warning(f"Failed to reach {b_id} during federated search.")
                
        # Sort results by timestamp (assuming HH:MM:SS format string)
        all_results.sort(key=lambda x: x.get('timestamp', ''))
        return all_results

# Flask Endpoints

node: BuildingNode = None

@app.route('/local_event', methods=['POST'])
def handle_local_event():
    data = request.json
    node.add_event(data)
    logging.info(f"Local Event: {data}")
    return jsonify({"status": "stored"}), 201

@app.route('/heartbeat', methods=['POST'])
def handle_heartbeat():
    data = request.json
    b_id = data.get("building")
    if b_id:
        node.receive_heartbeat(b_id)
    return jsonify({"status": "received"}), 200

@app.route('/search', methods=['GET'])
def handle_local_search():
    query = request.args.get('q', '')
    results = node.local_search(query)
    return jsonify({"results": results}), 200

@app.route('/federated_search', methods=['GET'])
def handle_federated_search():
    query = request.args.get('q', '')
    results = node.federated_search(query)
    return jsonify({
        "results": results, 
        "total": len(results),
        "participating_nodes": [node.building_id] + list(node.online_peers.keys())
    }), 200

def main():
    global node
    parser = argparse.ArgumentParser(description="Building Aggregator Node")
    parser.add_argument("--building-id", required=True, help="e.g., B0, B1, B2")
    args = parser.parse_args()

    # Initialize node
    node = BuildingNode(args.building_id)
    logging.info(f"Starting {node.building_id} Aggregator on port {node.port}")

    # Start background heartbeat thread
    hb_thread = threading.Thread(target=node.heartbeat_loop, daemon=True)
    hb_thread.start()

    # Run Flask server
    app.run(host='127.0.0.1', port=node.port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
