import os
import numpy as np
from keras_facenet import FaceNet
import cv2

embedder = FaceNet()

# Get backend folder (one level up from this script)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

cropped_path = os.path.join(BASE_DIR, "cropped")  # e.g. backend/cropped
output_path = os.path.join(BASE_DIR, "embeddings")  # e.g. backend/embeddings

if not os.path.exists(cropped_path):
    print(f"Cropped folder not found: {cropped_path}")
    exit()

data = {"names": [], "embeddings": []}

for person in os.listdir(cropped_path):
    person_folder = os.path.join(cropped_path, person)
    if not os.path.isdir(person_folder):
        continue

    for img_name in os.listdir(person_folder):
        if img_name.endswith(".jpg"):
            img_path = os.path.join(person_folder, img_name)
            print(f"Processing: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read {img_path}")
                continue

            embedding = embedder.embeddings([img])[0]
            data["names"].append(person)
            data["embeddings"].append(embedding)

# Save each embedding as <name>.npy inside backend/embeddings
os.makedirs(output_path, exist_ok=True)
for name, embedding in zip(data["names"], data["embeddings"]):
    np.save(os.path.join(output_path, f"{name}.npy"), embedding)
    print(f"Saved: {name}.npy")

print("All embeddings saved individually!")
