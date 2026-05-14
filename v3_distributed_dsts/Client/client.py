import socket
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", required=True, help="Entry node to query")
    parser.add_argument("--occupant", required=True, help="Occupant ID to find")
    args = parser.parse_args()
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'shared_config.json')
    with open(config_path, 'r') as f:
        network_config = json.load(f)['buildings']
    host, port = network_config[args.node]['host'], network_config[args.node]['port']
    
    print(f"Connecting to {args.node} to find {args.occupant}...")
    req = {"type": "SEARCH_REQ", "occupant_id": args.occupant, "federated": True}
    
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
            
            # Find the global maximum probability
            best_match = max(results, key=lambda x: x.get('prob', 0.0))
            
            print("\n=== DSTS FEDERATED SEARCH RESULT ===")
            print(f"Query: {args.occupant}")
            if best_match['prob'] >= 0.5:
                print(f"Location: Building {best_match['building']}, {best_match['zone']}")
                print(f"Confidence: {best_match['prob']:.3f} (>= 0.5)")
            else:
                print("Status: UNKNOWN")
                print(f"Highest confidence was {best_match['prob']:.3f} at Building {best_match['building']}, {best_match['zone']} (Below threshold)")
            
            print("\nNode Breakdown:")
            for r in results:
                print(f" - B{r['building']}: {r['zone']} (p={r['prob']:.3f})")
            
    except Exception as e:
        print(f"Search failed: {e}")

if __name__ == "__main__":
    main()
