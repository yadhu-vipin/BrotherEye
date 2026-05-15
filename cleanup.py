"""
cleanup.py
==========
Removes all generated runtime artifacts from building directories.
Run from the project root: python cleanup.py

Targets:
  - *_track.png, *_track_visual.png   (track plot images)
  - occupant_*_history.json           (per-occupant event logs)
  - event_history.json                (building event logs)
  - state_history.json                (state transition snapshots)
  - state_transition_tables.txt       (formatted state tables)
  - teleportation_log.txt             (teleportation detection logs)
  - search_result_*.json              (query result files)
  - queries/                          (timestamped query folders)
  - __pycache__/                      (Python bytecode cache)
"""

import os
import shutil
import glob
import sys

# Directories to scan
SCAN_DIRS = [
    os.path.join(os.path.dirname(__file__), "v5_final"),
    os.path.join(os.path.dirname(__file__), "live_simulation"),
]

# File patterns to delete (matched against filename)
JUNK_PATTERNS = [
    "*_track.png",
    "*_track_visual.png",
    "occupant_*_history.json",
    "event_history.json",
    "state_history.json",
    "state_transition_tables.txt",
    "teleportation_log.txt",
    "search_result_*.json",
]

# Directory names to remove entirely
JUNK_DIRS = [
    "__pycache__",
    "queries",
]


def cleanup():
    deleted_files = 0
    deleted_dirs = 0
    freed_bytes = 0

    for scan_dir in SCAN_DIRS:
        if not os.path.exists(scan_dir):
            continue

        # Walk all subdirectories
        for root, dirs, files in os.walk(scan_dir):
            # Delete matching files
            for pattern in JUNK_PATTERNS:
                for match in glob.glob(os.path.join(root, pattern)):
                    size = os.path.getsize(match)
                    os.remove(match)
                    freed_bytes += size
                    deleted_files += 1
                    print(f"  DEL  {os.path.relpath(match, os.path.dirname(__file__))}")

            # Delete matching directories
            for dname in JUNK_DIRS:
                dpath = os.path.join(root, dname)
                if os.path.isdir(dpath):
                    size = sum(
                        os.path.getsize(os.path.join(dp, f))
                        for dp, _, fnames in os.walk(dpath)
                        for f in fnames
                    )
                    shutil.rmtree(dpath)
                    freed_bytes += size
                    deleted_dirs += 1
                    print(f"  DEL  {os.path.relpath(dpath, os.path.dirname(__file__))}/")

    print()
    print("=" * 60)
    print(f"Cleanup complete.")
    print(f"  Files deleted:       {deleted_files}")
    print(f"  Directories deleted: {deleted_dirs}")
    print(f"  Space freed:         {freed_bytes / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    print("BrotherEye Cleanup")
    print("=" * 60)
    print("This will delete all generated runtime files from:")
    for d in SCAN_DIRS:
        print(f"  - {d}")
    print()

    if "--yes" not in sys.argv:
        ans = input("Proceed? (y/N): ").strip().lower()
        if ans != "y":
            print("Aborted.")
            sys.exit(0)

    print()
    cleanup()
