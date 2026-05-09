import json
import numpy as np
import face_recognition
from sklearn.datasets import fetch_lfw_people
import os
import shutil

IMAGES_PER_PERSON = 40

def augment_encodings(encodings, target_count):
    augmented = list(encodings)
    while len(augmented) < target_count:
        base = encodings[np.random.randint(0, len(encodings))]
        noise = np.random.normal(0, 0.01, base.shape)
        augmented.append(base + noise)
    return augmented[:target_count]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print("Fetching LFW Dataset...")
    lfw = fetch_lfw_people(min_faces_per_person=15, resize=0.4, color=True)

    all_occupants = []
    for b in ["B0", "B1", "B2", "B3", "B4"]:
        for i in range(1, 6):
            all_occupants.append(f"{b}_Person_{i}")

    reference_db = {}
    test_db      = {}
    idx = 0
    print("Generating face encodings...")
    for person_name in lfw.target_names:
        if idx >= 25:
            break
        lfw_idx = np.where(lfw.target_names == person_name)[0][0]
        img_idxs = np.where(lfw.target == lfw_idx)[0]

        encs = []
        for ii in img_idxs[:IMAGES_PER_PERSON]:
            img = (lfw.images[ii] * 255).astype(np.uint8)
            found = face_recognition.face_encodings(img)
            if found:
                encs.append(found[0])

        if encs:
            if len(encs) < IMAGES_PER_PERSON:
                encs = augment_encodings(encs, IMAGES_PER_PERSON)
            
            sid = all_occupants[idx]
            
            # Split: first 20 for Reference, remaining for Test
            mid = IMAGES_PER_PERSON // 2
            reference_db[sid] = [e.tolist() for e in encs[:mid]]
            test_db[sid]      = [e.tolist() for e in encs[mid:]]
            
            print(f"  {idx+1:>2}. {sid} ({person_name}) [20 ref, 20 test]")
            idx += 1

    # Save master copies
    ref_master = os.path.join(base_dir, 'reference_db.json')
    tst_master = os.path.join(base_dir, 'test_db.json')
    
    with open(ref_master, 'w') as f:
        json.dump(reference_db, f)
    with open(tst_master, 'w') as f:
        json.dump(test_db, f)
    
    print(f"\nMaster DBs saved to {base_dir}")

    # Copy into each building folder
    for b in ["B0", "B1", "B2", "B3", "B4"]:
        ref_dest = os.path.join(base_dir, f"Building_{b}", "reference_db.json")
        tst_dest = os.path.join(base_dir, f"Building_{b}", "test_db.json")
        
        shutil.copy2(ref_master, ref_dest)
        shutil.copy2(tst_master, tst_dest)
        print(f"  Copied to Building_{b}/")

    print("\nDone! All building folders now have their encodings DB.")

if __name__ == "__main__":
    main()
