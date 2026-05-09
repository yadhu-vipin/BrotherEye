import json
import numpy as np
import face_recognition
from sklearn.datasets import fetch_lfw_people
import os

IMAGES_PER_PERSON = 40

def augment_encodings(encodings, target_count):
    augmented = list(encodings)
    while len(augmented) < target_count:
        base = encodings[np.random.randint(0, len(encodings))]
        noise = np.random.normal(0, 0.01, base.shape)
        augmented.append(base + noise)
    return augmented[:target_count]

def main():
    print("Fetching LFW Dataset... This will take a few minutes.")
    lfw_data = fetch_lfw_people(min_faces_per_person=15, resize=0.4, color=True)
    
    encodings_db = {}
    person_idx = 0
    all_occupants = []
    for b in ["B0", "B1", "B2", "B3", "B4"]:
        for i in range(1, 6):
            all_occupants.append(f"{b}_Person_{i}")
            
    print("Generating Encodings...")
    for person_name in lfw_data.target_names:
        if person_idx >= 25: break
        idx_in_lfw = np.where(lfw_data.target_names == person_name)[0][0]
        image_indices = np.where(lfw_data.target == idx_in_lfw)[0]
        enc_list = []
        for idx in image_indices[:IMAGES_PER_PERSON]:
            img_uint8 = (lfw_data.images[idx] * 255).astype(np.uint8)
            encs = face_recognition.face_encodings(img_uint8)
            if encs: enc_list.append(encs[0])
                
        if len(enc_list) > 0:
            if len(enc_list) < IMAGES_PER_PERSON:
                enc_list = augment_encodings(enc_list, IMAGES_PER_PERSON)
            sys_id = all_occupants[person_idx]
            encodings_db[sys_id] = [e.tolist() for e in enc_list]
            print(f"Generated {IMAGES_PER_PERSON} encodings for {sys_id}")
            person_idx += 1

    db_path = os.path.join(os.path.dirname(__file__), 'shared_encodings_db.json')
    with open(db_path, 'w') as f:
        json.dump(encodings_db, f)
    print(f"Saved DB to {db_path}")

if __name__ == "__main__":
    main()
