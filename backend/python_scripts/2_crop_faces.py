import os
from mtcnn import MTCNN
import cv2
import sys

def crop_faces(input_dir, output_dir):
    detector = MTCNN()
    os.makedirs(output_dir, exist_ok=True)

    print(f"Looking inside: {input_dir}")
    for file in os.listdir(input_dir):
        if file.endswith(".jpg"):
            img_path = os.path.join(input_dir, file)
            print(f"Processing {img_path}")
            image = cv2.imread(img_path)
            if image is None:
                print(f"Could not read image: {img_path}")
                continue

            result = detector.detect_faces(image)
            if result:
                x, y, width, height = result[0]['box']
                x, y = abs(x), abs(y)
                cropped_face = image[y:y+height, x:x+width]
                out_path = os.path.join(output_dir, file)
                cv2.imwrite(out_path, cropped_face)
                print(f"Saved cropped face: {out_path}")
            else:
                print(f"No face detected in {file}")

# ✅ Get name from command line
if len(sys.argv) < 2:
    print("Name argument missing.")
    sys.exit(1)

name = sys.argv[1]

# ✅ Build Windows-style paths
base_dir = r"C:\Users\HP\face-main\backend"
input_dir = os.path.join(base_dir, "dataset", name)
output_dir = os.path.join(base_dir, "cropped", name)

if not os.path.exists(input_dir):
    print(f"Folder not found: {input_dir}")
else:
    crop_faces(input_dir, output_dir)
