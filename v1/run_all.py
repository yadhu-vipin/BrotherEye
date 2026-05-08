import subprocess
import time
import os
import sys

def main():
    print("Starting Distributed Event System Simulation...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    building_script = os.path.join(script_dir, "building_node.py")
    occupant_script = os.path.join(script_dir, "occupant_laptop.py")

    processes = []

    # Start 5 Building Aggregators
    buildings = ["B0", "B1", "B2", "B3", "B4"]
    for b in buildings:
        print(f"Starting Node {b}...")
        p = subprocess.Popen([sys.executable, building_script, "--building-id", b],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
    
    # Give servers a moment to start and exchange initial heartbeats
    print("Waiting 3 seconds for network to initialize...")
    time.sleep(3)

    # Start 25 Occupant Laptops (5 per building)
    for b in buildings:
        for i in range(1, 6):
            occupant_id = f"{b}_Person_{i}"
            print(f"Starting Occupant Simulator: {occupant_id}")
            p = subprocess.Popen([sys.executable, occupant_script, "--building-id", b, "--occupant-id", occupant_id],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(p)

    print("\nAll systems are running in the background.")
    print("You can now run searches using client.py in a separate terminal.")
    print("Example: python client.py --node B0 --query ENTRY")
    print("\nPress Ctrl+C to stop all simulation processes.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all processes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        print("All processes stopped.")

if __name__ == "__main__":
    main()
