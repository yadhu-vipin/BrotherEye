import argparse
import json
import os
import requests

def main():
    parser = argparse.ArgumentParser(description="Federated Search Client")
    parser.add_argument("--node", required=True, help="Building ID to query (e.g., B0, B1)")
    parser.add_argument("--query", required=True, help="Search term (e.g., Person_1, ENTRY, z1_b0)")
    args = parser.parse_args()

    # Load config to find node url
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        network_config = json.load(f)['buildings']
        
    if args.node not in network_config:
        print(f"Error: Unknown node {args.node}")
        return

    node_url = network_config[args.node]
    
    print(f"Connecting to node {args.node} ({node_url})...")
    print(f"Executing federated search for: '{args.query}'\n")

    try:
        resp = requests.get(f"{node_url}/federated_search", params={"q": args.query})
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("results", [])
            total = data.get("total", 0)
            participating_nodes = data.get("participating_nodes", [])
            
            print(f"--- Search Results ({total} found) ---")
            for res in results:
                print(f"[{res.get('timestamp')}] {res.get('occupant')} performed {res.get('event')} in {res.get('zone')} (Log from: {res.get('building')})")
            
            print(f"\nParticipating Nodes: {', '.join(participating_nodes)}")
        else:
            print(f"Search failed with status: {resp.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not connect to node {args.node}")
        print(str(e))

if __name__ == "__main__":
    main()
