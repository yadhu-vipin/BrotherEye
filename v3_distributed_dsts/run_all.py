import subprocess
import time
import os
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, 'shared_encodings_db.json')
    
    if not os.path.exists(db_path):
        print("Initial setup: Generating LFW Face Encodings DB...")
        subprocess.run([sys.executable, os.path.join(base_dir, 'generate_db.py')])
        
    print("\nStarting V3 DSTS Socket Architecture...")
    processes = []
    buildings = ["B0", "B1", "B2", "B3", "B4"]
    
    for b in buildings:
        b_dir = os.path.join(base_dir, f"Building_{b}")
        print(f"Starting {b} BSTS Node...")
        p = subprocess.Popen([sys.executable, "building_node.py", "--building-id", b], cwd=b_dir)
        processes.append(p)
        
    time.sleep(3)
    
    for b in buildings:
        for i in range(1, 6):
            occ_id = f"{b}_Person_{i}"
            occ_dir = os.path.join(base_dir, f"Laptop_{occ_id}")
            print(f"Starting camera laptop {occ_id}...")
            p = subprocess.Popen([sys.executable, "occupant_laptop.py", "--building-id", b, "--occupant-id", occ_id], cwd=occ_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(p)
            
    print("\nAll 30 processes running in background. DSTS logic is active.")
    print("Use Client folder to run federated probability searches. Press Ctrl+C to stop.")
    
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        for p in processes: p.terminate()
        print("Stopped.")

if __name__ == "__main__":
    main()
