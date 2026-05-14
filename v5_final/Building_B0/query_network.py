import socket
import json
import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Query the distributed DSTS network for location history or verify zone connections.")
    parser.add_argument("--person", required=False,
                        help="Occupant ID to search for (e.g. B1_Person_2)")
    parser.add_argument("--check_connection", nargs=2, metavar=('ZONE1', 'ZONE2'),
                        help="Verify physical connection/adjacency between two zones (e.g. zT_B0 z1_B0)")
    args = parser.parse_args()

    if not args.person and not args.check_connection:
        parser.error("You must provide either --person or --check_connection")

    folder   = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(folder, 'config.json')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    bid  = cfg['my_building_id']
    host = cfg['buildings'][bid]['host']
    port = cfg['buildings'][bid]['port']

    if args.check_connection:
        z1, z2 = args.check_connection
        req = {"type": "CHECK_CONNECTION", "zone1": z1, "zone2": z2}
        print(f"Verifying physical connectivity between '{z1}' and '{z2}' via Building {bid} ({host}:{port})...")
        print()
    else:
        req = {"type": "SEARCH_REQ", "occupant_id": args.person, "federated": True}
        print(f"Querying network for '{args.person}' via Building {bid} ({host}:{port})...")
        print()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(8.0)
            s.connect((host, int(port)))
            s.sendall((json.dumps(req) + "\n").encode('utf-8'))

            buf = b""
            while True:
                chunk = s.recv(16384)
                if not chunk:
                    break
                buf += chunk
                if b"\n" in buf:
                    break

            resp = json.loads(buf.decode('utf-8').strip())

    except Exception as e:
        print(f"ERROR: Could not reach building server — {e}")
        print("Make sure building_server.py is running first.")
        sys.exit(1)

    if args.check_connection:
        z1, z2 = args.check_connection
        connected = resp.get("connected", False)
        print("=" * 80)
        print(f"CONNECTIVITY VERIFICATION: {z1} <-> {z2}")
        print("=" * 80)
        if connected:
            print(f"Result: TRUE — Zones '{z1}' and '{z2}' are physically adjacent/connected.")
        else:
            print(f"Result: FALSE — Zones '{z1}' and '{z2}' are NOT physically connected.")
        print()
        sys.exit(0)

    results   = resp.get("results", [])
    snapshots = resp.get("state_snapshots", [])

    # ── Print tracking timeline ──────────────────────────────────────────────
    print("=" * 80)
    print(f"TRACK: {args.person}")
    print("=" * 80)
    print()

    if results:
        print(f"{'Time':<12} {'Building':<10} {'Zone':<16} {'Probability':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r['timestamp']:<12} {r['building']:<10} "
                  f"{r['zone']:<16} {r['prob']:<12.4f}")
        print("-" * 80)
        print(f"Total events: {len(results)}")
    else:
        print("No tracking events found for this person.")

    # ── Print current state table snapshot ────────────────────────────────────
    if snapshots:
        print()
        print("=" * 80)
        print(f"CURRENT STATE TABLE SNAPSHOT")
        print("=" * 80)
        print()
        for snap in snapshots:
            if not snap:
                continue
            print(f"  Building {snap['building']}:")
            for zone, prob in snap['zone_probabilities'].items():
                marker = " *" if prob >= 0.5 else ""
                print(f"    {zone:<20} {prob:.4f}{marker}")
            print(f"    >> Current location: {snap['current_zone']} "
                  f"(p={snap['current_prob']:.4f})")
            print()

    # ── Save to JSON ─────────────────────────────────────────────────────────
    out = os.path.join(folder, f"search_result_{args.person}.json")
    report = {
        "query": args.person,
        "queried_from": bid,
        "tracking_events": results,
        "state_snapshots": snapshots
    }
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to: {out}")

if __name__ == "__main__":
    main()
