import subprocess
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
            
    print("\nAll 30 processes running in background.")
    print("Use Client folder to run searches. Press Ctrl+C to stop.")
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        for p in processes: p.terminate()
        print("Stopped.")

if __name__ == "__main__":
    main()
