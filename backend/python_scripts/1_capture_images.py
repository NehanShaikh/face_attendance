import cv2
import os
import sys

if len(sys.argv) < 2:
    print("Name argument missing")
    sys.exit(1)

name = sys.argv[1]

save_dir = f"dataset/{name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("Press 's' to save an image. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1)
    if key == ord('s'):
        img_path = f"{save_dir}/{name}_{count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        count += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
