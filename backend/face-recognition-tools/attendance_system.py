import cv2
import numpy as np
from keras_facenet import FaceNet
import requests
import psycopg2
import os
import sys
from datetime import datetime
import pytz

# --------------------------
# Fix Unicode printing on Windows
# --------------------------
sys.stdout.reconfigure(encoding='utf-8')

# --------------------------
# Config - UPDATED FOR RENDER
# --------------------------
USE_COSINE = False          # True to use cosine similarity
THRESHOLD = 1.0             # Euclidean: ~0.9–1.2 | Cosine: ~0.5–0.7
FRAMES_REQUIRED = 10        # Number of frames before marking attendance
ATTENDANCE_API = "https://face-attendance-9vis.onrender.com/attendance"
TIMEZONE = 'Asia/Kolkata'   # Ensure correct timezone

# --------------------------
# Database Configuration - HARDCODED FOR LOCAL USE
# --------------------------
DB_CONFIG = {
    "host": "dpg-d3h73lhr0fns73c5cesg-a.oregon-postgres.render.com",
    "port": 5432,
    "database": "face_db_fym1",
    "user": "face_db_fym1_user",
    "password": "XyD8oZRvbMjx6o5XKmJoFYKjaPWz2uOV"
}

# --------------------------
# Initialize FaceNet
# --------------------------
embedder = FaceNet()
already_marked = set()
detection_counts = {}  # Track detections per student

# --------------------------
# PostgreSQL connection - UPDATED
# --------------------------
try:
    conn = psycopg2.connect(**DB_CONFIG)
    print("✅ PostgreSQL connection successful")
except Exception as e:
    print("❌ PostgreSQL connection error:", e)
    exit()

cursor = conn.cursor()

# --------------------------
# Load saved embeddings
# --------------------------
cursor.execute("""
    SELECT s.student_id, s.name, f.embedding
    FROM face_embeddings f
    JOIN students s ON s.student_id = f.student_id
""")

known_embeddings = []
known_names = []
known_ids = []

for student_id, name, emb_data in cursor.fetchall():
    if emb_data:
        try:
            if isinstance(emb_data, memoryview):
                emb_data = emb_data.tobytes()
            elif isinstance(emb_data, str):
                emb_data = bytes.fromhex(emb_data[2:]) if emb_data.startswith("\\x") else emb_data.encode()
            emb = np.frombuffer(emb_data, dtype=np.float32)
            known_embeddings.append(emb)
            known_names.append(name)
            known_ids.append(student_id)
        except Exception as e:
            print(f"❌ Failed to parse embedding for {name}: {e}")
    else:
        print(f"⚠️ No embedding found for {name}")

print(f"✅ Loaded {len(known_names)} embeddings from database")

# --------------------------
# Load student timetables into memory
# --------------------------
cursor.execute("""
    SELECT st.student_id, t.day, t.start_time, t.end_time, s.name AS subject
    FROM student_timetable st
    JOIN timetable t ON st.timetable_id = t.timetable_id
    JOIN subjects s ON t.subject_id = s.subject_id
""")

# Dictionary: student_id -> list of timetable entries
timetable_dict = {}
for student_id, day, start_time, end_time, subject in cursor.fetchall():
    timetable_dict.setdefault(student_id, []).append({
        "day": day.strip().lower(),  # strip spaces & lowercase
        "start": start_time,
        "end": end_time,
        "subject": subject
    })

print(f"✅ Loaded timetable for {len(timetable_dict)} students")

# --------------------------
# Similarity functions
# --------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --------------------------
# Check if student has class right now
# --------------------------
def has_class_now(student_id):
    if student_id not in timetable_dict:
        print(f"❌ No timetable found for student_id: {student_id}")
        return False, None

    ist = pytz.timezone(TIMEZONE)
    now = datetime.now(ist)
    current_day = now.strftime("%A").strip().lower()
    current_time = now.time()

    print(f"🎯 DEBUG - Current time: {current_day} {current_time}")

    for entry in timetable_dict[student_id]:
        entry_day = entry["day"].strip().lower()
        
        # Convert times to comparable format
        if isinstance(entry["start"], str):
            start_time = datetime.strptime(entry["start"], "%H:%M:%S").time()
        else:
            start_time = entry["start"]
            
        if isinstance(entry["end"], str):
            end_time = datetime.strptime(entry["end"], "%H:%M:%S").time()
        else:
            end_time = entry["end"]

        print(f"🎯 DEBUG - Checking: {entry_day} {start_time}-{end_time}")

        # Debug the comparison
        day_match = entry_day == current_day
        time_in_range = start_time <= current_time <= end_time
        
        print(f"🎯 DEBUG - Day match: {day_match}, Time in range: {time_in_range}")

        if day_match and time_in_range:
            print(f"✅ Class confirmed: {entry['subject']}")
            return True, entry["subject"]

    print("❌ No matching class found")
    return False, None

# --------------------------
# Attendance marking function
# --------------------------
def mark_attendance(name):
    if name not in already_marked:
        try:
            print(f"📤 Sending attendance request for: {name}")
            r = requests.post(ATTENDANCE_API, json={"name": name}, timeout=10)
            if r.ok:
                already_marked.add(name)
                response_data = r.json()
                print(f"✅ Attendance marked successfully: {name}")
                print(f"📨 Response: {response_data}")
            else:
                error_msg = r.text if r.text else "No error message"
                print(f"❌ API failed ({r.status_code}): {error_msg}")
        except requests.exceptions.Timeout:
            print(f"❌ Request timeout for {name}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Connection error - Is the server running at {ATTENDANCE_API}?")
        except Exception as e:
            print(f"❌ Request error for {name}: {e}")

# --------------------------
# Display startup information
# --------------------------
print(f"\n🔧 Configuration:")
print(f"   - Timezone: {TIMEZONE}")
print(f"   - Frames required: {FRAMES_REQUIRED}")
print(f"   - Threshold: {THRESHOLD}")
print(f"   - Similarity: {'Cosine' if USE_COSINE else 'Euclidean'}")
print(f"   - Students loaded: {len(known_names)}")
print(f"   - Timetable entries: {sum(len(v) for v in timetable_dict.values())}")
print(f"   - API Endpoint: {ATTENDANCE_API}")
print("\n🚀 Starting face recognition...\n")

# --------------------------
# Webcam loop
# --------------------------
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()

print("✅ Camera opened successfully")
print("🔑 Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from camera")
        break

    try:
        results = embedder.extract(frame, threshold=0.95)
    except Exception as e:
        print("❌ Face extraction failed:", e)
        continue

    if not results:
        # Show "No faces detected" on frame
        cv2.putText(frame, "No faces detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        for res in results:
            box = res.get('box')
            emb = res.get('embedding')

            if box is None or emb is None:
                continue

            x, y, w, h = box
            x, y = max(0, x), max(0, y)

            name = "Unknown"
            student_id = None
            confidence = 0

            # Compare embeddings
            if USE_COSINE:
                max_sim, best_match_idx = -1, None
                for i, known_emb in enumerate(known_embeddings):
                    sim = cosine_similarity(emb, known_emb)
                    if sim > max_sim:
                        max_sim, best_match_idx = sim, i
                if max_sim >= THRESHOLD:
                    name = known_names[best_match_idx]
                    student_id = known_ids[best_match_idx]
                    confidence = max_sim
            else:
                min_dist, best_match_idx = float('inf'), None
                for i, known_emb in enumerate(known_embeddings):
                    dist = np.linalg.norm(emb - known_emb)
                    if dist < min_dist:
                        min_dist, best_match_idx = dist, i
                if min_dist <= THRESHOLD:
                    name = known_names[best_match_idx]
                    student_id = known_ids[best_match_idx]
                    confidence = 1 - (min_dist / THRESHOLD)  # Convert to confidence score

            # Draw rectangle and label
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Frame-count based attendance
            if name != "Unknown":
                detection_counts[name] = detection_counts.get(name, 0) + 1
                current_count = detection_counts[name]
                
                if current_count >= FRAMES_REQUIRED and name not in already_marked:
                    # Check timetable before marking
                    has_class, subject = has_class_now(student_id)
                    if has_class:
                        print(f"🎯 Attempting to mark attendance for {name} (detections: {current_count}/{FRAMES_REQUIRED})")
                        mark_attendance(name)
                    else:
                        print(f"⏰ No class now for {name}, not marking attendance")
                    detection_counts[name] = 0
                elif name not in already_marked:
                    # Show progress every 5 frames to avoid spam
                    if current_count % 5 == 0:
                        print(f"📊 {name}: {current_count}/{FRAMES_REQUIRED} detections")
            else:
                detection_counts["Unknown"] = 0

    # Display status information on frame
    status_text = f"Marked: {len(already_marked)} | Press 'q' to quit"
    cv2.putText(frame, status_text, (10, frame.shape[0] - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Face Recognition Attendance", frame)
    
    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n🛑 Quitting...")
        break

# --------------------------
# Cleanup
# --------------------------
cap.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
print("✅ Cleanup done, program exited.")
print(f"📊 Summary: Marked attendance for {len(already_marked)} students")
if already_marked:
    print(f"🎯 Students marked: {', '.join(already_marked)}")
