import face_recognition
import numpy as np
from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
print("Libraries imported successfully!")
lfw_data = fetch_lfw_people(min_faces_per_person=1, resize=0.4, color=True)
print(f"LFW loaded: {len(lfw_data.images)} images")
