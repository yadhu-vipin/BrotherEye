import os
import glob
import re

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    out_lines = []
    in_handle_with_conn = False
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 1. db_path
        if "db_path    = os.path.join(self.folder, 'encodings_db.json')" in line:
            line = line.replace("encodings_db.json", "reference_db.json")

        # 2. timeout_threshold
        if "self.timeout_threshold = 30.0" in line:
            line = line.replace("30.0", "10.0")

        # 3. state_lock
        if "self.plot_lock         = threading.Lock()" in line:
            # check if next line is already state_lock
            if i + 1 < len(lines) and "self.state_lock" not in lines[i+1]:
                line = line + "        self.state_lock        = threading.Lock()\n"

        out_lines.append(line)
        i += 1

    content = "".join(out_lines)
    
    # Now string replacements for the logic
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

    # Indent everything inside "with conn:" inside "def handle(self, conn, addr):"
    handle_def = "    def handle(self, conn, addr):\n        with conn:\n"
    if handle_def in content and "try:" not in content.split(handle_def)[1][:20]:
        parts = content.split(handle_def)
        pre = parts[0]
        rest = parts[1]
        
        # We need to find the end of handle function. It ends when we hit "    # ── main loop" or similar.
        end_idx = rest.find("    # ── main loop")
        if end_idx == -1:
            end_idx = rest.find("    def start(self):")
        
        handle_body = rest[:end_idx]
        post = rest[end_idx:]
        
        # Add 4 spaces to every line in handle_body that is not empty
        indented_body = []
        for line in handle_body.split('\\n'):
            if line.strip():
                indented_body.append("    " + line)
            else:
                indented_body.append(line)
                
        new_handle_body = "\\n".join(indented_body)
        
        # Add the try/except around it
        try_except_body = "            try:\\n" + new_handle_body
        
        # Remove trailing blank lines before adding except
        try_except_body = try_except_body.rstrip() + "\\n"
        try_except_body += '''            except Exception as e:
                logging.error(f"Error handling request from {addr}: {e}")
                import traceback
                traceback.print_exc()

'''
        content = pre + handle_def + try_except_body + post

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
