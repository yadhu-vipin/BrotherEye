import socket
import json
import os
import sys
import argparse
import base64
from datetime import datetime


def recv_full(sock):
    """Receive a full JSON message terminated by newline."""
    buf = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        buf += chunk
        if b"\n" in buf:
            break
    text = buf.decode('utf-8').strip()
    return json.loads(text) if text else None


def send_request(host, port, payload, timeout=10.0):
    """Send a TCP request and return the JSON response."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect((host, int(port)))
            s.sendall((json.dumps(payload) + "\n").encode('utf-8'))
            return recv_full(s)
    except Exception as e:
        return None


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

    # ── Connection check mode ─────────────────────────────────────────────────
    if args.check_connection:
        z1, z2 = args.check_connection
        req = {"type": "CHECK_CONNECTION", "zone1": z1, "zone2": z2}
        print(f"Verifying physical connectivity between '{z1}' and '{z2}' via Building {bid} ({host}:{port})...")
        print()

        resp = send_request(host, port, req)
        if not resp:
            print("ERROR: Could not reach building server.")
            sys.exit(1)

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

    # ── Person search mode ────────────────────────────────────────────────────
    person = args.person
    print(f"Querying network for '{person}' via Building {bid} ({host}:{port})...")
    print()

    # Step 1: Federated SEARCH_REQ to get tracking events + state snapshots
    search_req = {"type": "SEARCH_REQ", "occupant_id": person, "federated": True}
    search_resp = send_request(host, port, search_req)

    if not search_resp:
        print("ERROR: Could not reach building server — make sure building_server.py is running first.")
        sys.exit(1)

    results   = search_resp.get("results", [])
    snapshots = search_resp.get("state_snapshots", [])

    # Step 2: OCCUPANT_DATA_REQ to local building
    data_req = {"type": "OCCUPANT_DATA_REQ", "occupant_id": person}
    local_data = send_request(host, port, data_req)

    # Step 3: OCCUPANT_DATA_REQ to all online peers
    peer_data = {}
    for b_id, info in cfg['buildings'].items():
        if b_id == bid:
            continue
        resp = send_request(info['host'], info['port'], data_req, timeout=5.0)
        if resp and resp.get("type") == "OCCUPANT_DATA_RES":
            peer_data[b_id] = resp

    # ── Create timestamped query folder ───────────────────────────────────────
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    query_folder_name = f"query_{person}_{timestamp_str}"
    queries_dir = os.path.join(folder, "queries")
    query_folder = os.path.join(queries_dir, query_folder_name)
    os.makedirs(query_folder, exist_ok=True)

    # ── Print tracking timeline ───────────────────────────────────────────────
    print("=" * 80)
    print(f"TRACK: {person}")
    print("=" * 80)
    print()

    if results:
        print(f"{'Time':<12} {'Building':<10} {'Zone':<16} {'Probability':<12} {'Status':<20}")
        print("-" * 80)
        spurious_count = 0
        for r in results:
            is_spurious = r.get('spurious', False)
            status = "[TELEPORTATION NULLIFIED]" if is_spurious else "Valid"
            if is_spurious:
                spurious_count += 1
            print(f"{r['timestamp']:<12} {r['building']:<10} "
                  f"{r['zone']:<16} {r['prob']:<12.4f} {status}")
        print("-" * 80)
        print(f"Total events: {len(results)}")
        print(f"  Valid:    {len(results) - spurious_count}")
        print(f"  Spurious: {spurious_count} (teleportation rejected)")
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

    # ── Save everything to the timestamped folder ─────────────────────────────

    # 1. Full search report
    report = {
        "query": person,
        "queried_from": bid,
        "query_timestamp": timestamp_str,
        "tracking_events": results,
        "state_snapshots": snapshots
    }
    with open(os.path.join(query_folder, "search_result.json"), 'w') as f:
        json.dump(report, f, indent=2)

    # 2. Event history (combined from all buildings)
    all_event_history = []
    if local_data and local_data.get("event_history"):
        all_event_history.extend(local_data["event_history"])
    for b_id, pdata in peer_data.items():
        if pdata.get("event_history"):
            all_event_history.extend(pdata["event_history"])
    all_event_history.sort(key=lambda x: x.get("timestamp", ""))
    with open(os.path.join(query_folder, "event_history.json"), 'w') as f:
        json.dump(all_event_history, f, indent=2)

    # 3. Occupant-specific history (combined)
    occupant_history = []
    if local_data and local_data.get("occupant_history"):
        occupant_history.extend(local_data["occupant_history"])
    for b_id, pdata in peer_data.items():
        if pdata.get("occupant_history"):
            occupant_history.extend(pdata["occupant_history"])
    occupant_history.sort(key=lambda x: x.get("timestamp", ""))
    with open(os.path.join(query_folder, f"occupant_{person}_history.json"), 'w') as f:
        json.dump(occupant_history, f, indent=2)

    # 4. State transition history (per building)
    state_histories = {}
    if local_data and local_data.get("state_history"):
        state_histories[bid] = local_data["state_history"]
    for b_id, pdata in peer_data.items():
        if pdata.get("state_history"):
            state_histories[b_id] = pdata["state_history"]
    with open(os.path.join(query_folder, "state_history.json"), 'w') as f:
        json.dump(state_histories, f, indent=2)

    # 5. State table snapshot (formatted text)
    state_table_lines = []
    state_table_lines.append(f"STATE TABLE SNAPSHOT FOR: {person}")
    state_table_lines.append(f"Query Time: {timestamp_str}")
    state_table_lines.append(f"Queried From: Building {bid}")
    state_table_lines.append("=" * 80)
    for snap in snapshots:
        if not snap:
            continue
        state_table_lines.append(f"\n  Building {snap['building']}:")
        for zone, prob in snap['zone_probabilities'].items():
            marker = " *" if prob >= 0.5 else ""
            state_table_lines.append(f"    {zone:<20} {prob:.4f}{marker}")
        state_table_lines.append(f"    >> Current location: {snap['current_zone']} "
                                  f"(p={snap['current_prob']:.4f})")
    state_table_lines.append("\n" + "=" * 80)
    with open(os.path.join(query_folder, "state_table.txt"), 'w') as f:
        f.write("\n".join(state_table_lines))

    # 6. Track images (from each building that has one)
    saved_tracks = []
    if local_data and local_data.get("track_image_base64"):
        img_bytes = base64.b64decode(local_data["track_image_base64"])
        img_name = f"track_{bid}.png"
        with open(os.path.join(query_folder, img_name), 'wb') as f:
            f.write(img_bytes)
        saved_tracks.append(img_name)
    for b_id, pdata in peer_data.items():
        if pdata.get("track_image_base64"):
            img_bytes = base64.b64decode(pdata["track_image_base64"])
            img_name = f"track_{b_id}.png"
            with open(os.path.join(query_folder, img_name), 'wb') as f:
                f.write(img_bytes)
            saved_tracks.append(img_name)

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"QUERY RESULTS SAVED")
    print("=" * 80)
    print(f"  Folder: {query_folder}")
    print(f"  Contents:")
    print(f"    - search_result.json          (tracking events + state snapshots)")
    print(f"    - event_history.json          (all detection events from all buildings)")
    print(f"    - occupant_{person}_history.json  (person-specific events)")
    print(f"    - state_history.json          (full state transition log per building)")
    print(f"    - state_table.txt             (formatted state table snapshot)")
    if saved_tracks:
        for t in saved_tracks:
            print(f"    - {t:<30} (track visualization)")
    else:
        print(f"    - (no track images available)")
    print()


if __name__ == "__main__":
    main()
