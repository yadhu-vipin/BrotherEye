import socket
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
            s.sendall((json.dumps(req) + "\n").encode('utf-8'))
            
            buffer = ""
            while True:
                data = s.recv(4096)
                if not data: break
                buffer += data.decode('utf-8')
                if "\n" in buffer: break
                
            resp = json.loads(buffer.strip())
            
            results = resp.get("results", [])
            print(f"--- Search Results ({len(results)} found) ---")
            for res in results:
                print(f"[{res.get('timestamp')}] {res.get('occupant')} performed {res.get('event')} in {res.get('zone')} (Log from: {res.get('building')})")
            print(f"\nParticipating Nodes: {', '.join(resp.get('participating', []))}")
            
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    main()
