import argparse
import json
import os
import random
import time
from datetime import datetime
import requests

def main():
    parser = argparse.ArgumentParser(description="Occupant Laptop Simulation")
    parser.add_argument("--building-id", required=True, help="e.g., B0, B1")
    parser.add_argument("--occupant-id", required=True, help="e.g., Person_1")
    args = parser.parse_args()

    # Load config to find building aggregator port
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        network_config = json.load(f)['buildings']
        
    if args.building_id not in network_config:
        print(f"Error: Unknown building {args.building_id}")
        return

    aggregator_url = network_config[args.building_id]
    
    event_types = ["ENTRY", "MOVE", "EXIT"]
    # Simulated zones for the building
    zones = [f"z1_{args.building_id}", f"z2_{args.building_id}", f"z3_{args.building_id}", f"zT_{args.building_id}"]

    print(f"Starting simulation for {args.occupant_id} in {args.building_id}")
    print(f"Sending events to {aggregator_url}/local_event")

    try:
        while True:
            # Generate random event
            event_type = random.choice(event_types)
            zone = random.choice(zones)
            timestamp = datetime.now().strftime("%H:%M:%S")

            event_data = {
                "building": args.building_id,
                "occupant": args.occupant_id,
                "zone": zone,
                "event": event_type,
                "timestamp": timestamp
            }

            try:
                resp = requests.post(f"{aggregator_url}/local_event", json=event_data, timeout=2.0)
                if resp.status_code == 201:
                    print(f"[{timestamp}] Sent {event_type} at {zone}")
                else:
                    print(f"[{timestamp}] Error sending event: HTTP {resp.status_code}")
            except requests.exceptions.RequestException:
                print(f"[{timestamp}] Error: Could not connect to aggregator at {aggregator_url}")

            # Sleep for random duration between 5 to 15 seconds
            time.sleep(random.uniform(5.0, 15.0))
            
    except KeyboardInterrupt:
        print(f"\nSimulation stopped for {args.occupant_id}")

if __name__ == "__main__":
    main()
