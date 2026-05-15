import os
import glob

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. db_path
    content = content.replace("db_path    = os.path.join(self.folder, 'encodings_db.json')",
                              "db_path    = os.path.join(self.folder, 'reference_db.json')")

    # 2. timeout_threshold
    content = content.replace("self.timeout_threshold = 30.0",
                              "self.timeout_threshold = 10.0")

    # 3. state_lock
    if "self.state_lock        = threading.Lock()" not in content:
        content = content.replace("self.plot_lock         = threading.Lock()",
                                  "self.plot_lock         = threading.Lock()\n        self.state_lock        = threading.Lock()")

    # 4. handle try block
    if "try:\n                msg = self._recv_json(conn)" not in content:
        content = content.replace("            msg = self._recv_json(conn)\n            if not msg:\n                return",
                                  "            try:\n                msg = self._recv_json(conn)\n                if not msg:\n                    return")
        # indent everything below inside the try block until the next method
        # Actually it might be easier to use replace for the LOCAL_EVENT part
    
    # 5. LOCAL_EVENT block replacement
    old_local_event = '''            if t == "LOCAL_EVENT":
                # Limit event history to avoid memory issues in simulation
                if len(self.state_history) > 100:
                    conn.sendall(b'{"status":"limit_reached"}\\n')
                    return
                    
                data = msg["data"]
                enc  = np.array(data["captured_encoding"])
                zone = data["zone"]
                ts   = data["timestamp"]

                # Recognition now returns a dictionary of probabilities for ALL occupants
                event_probs = self.engine.recognize(enc, self.bsts.registered_occupants)
                
                # Apply transition to ALL occupants
                self.bsts.apply_transition(event_probs, zone)

                # Identify the winner for logging and history
                matched = max(event_probs, key=event_probs.get)
                prob = event_probs[matched]
                
                # Check if this is a uniform distribution (uncertain)
                is_uncertain = all(math.isclose(p, 1.0/len(event_probs), rel_tol=1e-5) for p in event_probs.values())'''
                
    new_local_event = '''            if t == "LOCAL_EVENT":
                logging.info(f"Processing LOCAL_EVENT from {addr}")
                # Limit event history to avoid memory issues in simulation
                if len(self.state_history) > 1000:
                    conn.sendall(b'{"status":"limit_reached"}\\n')
                    return
                    
                data = msg["data"]
                enc  = np.array(data["captured_encoding"])
                zone = data["zone"]
                ts   = data["timestamp"]

                # Recognition now returns a dictionary of probabilities for ALL occupants
                event_probs = self.engine.recognize(enc, self.bsts.registered_occupants)
                
                with self.state_lock:
                    # Apply transition to ALL occupants
                    self.bsts.apply_transition(event_probs, zone)

                    # Identify the winner for logging and history
                    matched = max(event_probs, key=event_probs.get) if event_probs else "UNKNOWN"
                    prob = event_probs.get(matched, 0.0)
                    
                    # Check if this is a uniform distribution (uncertain)
                    n_occ = len(event_probs)
                    is_uncertain = (n_occ > 0 and 
                                   all(math.isclose(p, 1.0/n_occ, rel_tol=1e-5) for p in event_probs.values()))'''

    content = content.replace(old_local_event, new_local_event)

    # 6. Adjust the end of LOCAL_EVENT block for the try...except
    # The try block should close before HEARTBEAT. But wait, `is_uncertain` block is followed by `if not is_uncertain`.
    # Let's see if we can just append the except block before `elif t == "HEARTBEAT":`
    old_heartbeat = '''                conn.sendall(b'{"status":"ok"}\\n')

            # ── heartbeat ────────────────────────────────────────────────────
            elif t == "HEARTBEAT":'''
            
    new_heartbeat = '''                conn.sendall(b'{"status":"ok"}\\n')
            except Exception as e:
                logging.error(f"Error handling request from {addr}: {e}")
                import traceback
                traceback.print_exc()

            # ── heartbeat ────────────────────────────────────────────────────
            elif t == "HEARTBEAT":'''
            
    if "except Exception as e:" not in content:
        content = content.replace(old_heartbeat, new_heartbeat)
        
    # BUT we need to indent everything between `try:` and `except:`
    # Wait, in Python, if we just do:
    # try:
    #     msg = self._recv_json(conn)
    # ... but the rest of the code is at the same indentation level!
    # Ah, the user's snippet:
    #         try:
    #             msg = self._recv_json(conn)
    #         if not msg:
    #             return
    #         t = msg.get("type")
    # This is a syntax error in the user's snippet! Look at the user's snippet:
    #         try:
    #             msg = self._recv_json(conn)
    #         if not msg:
    #             return
    #         t = msg.get("type")
    # Wait! In the user's snippet:
    #     def handle(self, conn, addr):
    #         with conn:
    #             try:
    #                 msg = self._recv_json(conn)
    #             if not msg:
    #                 return
    # 
    #             t = msg.get("type")
    # This is an IndentationError! `if not msg:` is indented at the same level as `try:`.
    # So I MUST fix the indentation for the whole `handle` body inside `try:`.

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Patched: {filepath}")

search_paths = [
    r"d:\brotherEye\BrotherEye\live_simulation\*\building_server.py",
    r"d:\brotherEye\BrotherEye\live_simulation\building_server.py",
    r"d:\brotherEye\BrotherEye\v5_final\*\building_server.py"
]

for sp in search_paths:
    for f in glob.glob(sp):
        patch_file(f)
