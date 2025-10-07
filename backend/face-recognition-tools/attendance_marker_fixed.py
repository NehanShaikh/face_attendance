import cv2
import numpy as np
import requests
import psycopg2
import os
import sys
from datetime import datetime
import pytz
import pickle

# --------------------------
# Fix Unicode printing on Windows
# --------------------------
sys.stdout.reconfigure(encoding='utf-8')

# --------------------------
# Config
# --------------------------
THRESHOLD = 0.6
FRAMES_REQUIRED = 10
ATTENDANCE_API = "https://face-attendance-9vis.onrender.com/attendance"
TIMEZONE = 'Asia/Kolkata'

# --------------------------
# Database Configuration
# --------------------------
DB_CONFIG = {
    "host": "dpg-d3h73lhr0fns73c5cesg-a.oregon-postgres.render.com",
    "port": 5432,
    "database": "face_db_fym1",
    "user": "face_db_fym1_user",
    "password": "XyD8oZRvbMjx6o5XKmJoFYKjaPWz2uOV"
}

# --------------------------
# Initialize Face Detector with proper path handling
# --------------------------
def load_face_detector():
    print("🚀 Loading face detection model...")
    
    # Try multiple possible paths for the cascade file
    possible_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_default.xml',
        os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                detector = cv2.CascadeClassifier(path)
                if not detector.empty():
                    print(f"✅ Face detector loaded from: {path}")
                    return detector
        except Exception as e:
            continue
    
    # If no file found, try to create a basic face detector using HOG
    print("⚠️ Using basic face detection (HOG)")
    return None

# --------------------------
# Global variables
# --------------------------
already_marked = set()
detection_counts = {}
known_faces = []
known_names = []
known_ids = []
face_detector = load_face_detector()

# --------------------------
# Database Connection
# --------------------------
def connect_database():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ PostgreSQL connection successful")
        return conn
    except Exception as e:
        print(f"❌ PostgreSQL connection error: {e}")
        return None

# --------------------------
# Load saved embeddings from database
# --------------------------
def load_embeddings():
    global known_faces, known_names, known_ids
    
    conn = connect_database()
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.student_id, s.name, f.embedding
            FROM face_embeddings f
            JOIN students s ON s.student_id = f.student_id
        """)

        for student_id, name, emb_data in cursor.fetchall():
            if emb_data:
                try:
                    if isinstance(emb_data, memoryview):
                        emb_data = emb_data.tobytes()
                    elif isinstance(emb_data, str):
                        emb_data = bytes.fromhex(emb_data[2:]) if emb_data.startswith("\\x") else emb_data.encode()
                    
                    emb = np.frombuffer(emb_data, dtype=np.float32)
                    known_faces.append(emb)
                    known_names.append(name)
                    known_ids.append(student_id)
                    print(f"✅ Loaded embedding for {name}")
                except Exception as e:
                    print(f"❌ Failed to parse embedding for {name}: {e}")
            else:
                print(f"⚠️ No embedding found for {name}")

        cursor.close()
        conn.close()
        
        print(f"✅ Loaded {len(known_names)} embeddings from database")
        return True
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False

# --------------------------
# Load student timetables
# --------------------------
def load_timetable():
    timetable_dict = {}
    
    conn = connect_database()
    if not conn:
        return timetable_dict
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT st.student_id, t.day, t.start_time, t.end_time, s.name AS subject
            FROM student_timetable st
            JOIN timetable t ON st.timetable_id = t.timetable_id
            JOIN subjects s ON t.subject_id = s.subject_id
        """)

        for student_id, day, start_time, end_time, subject in cursor.fetchall():
            timetable_dict.setdefault(student_id, []).append({
                "day": day.strip().lower(),
                "start": start_time,
                "end": end_time,
                "subject": subject
            })

        cursor.close()
        conn.close()
        print(f"✅ Loaded timetable for {len(timetable_dict)} students")
        
    except Exception as e:
        print(f"❌ Error loading timetable: {e}")
    
    return timetable_dict

# --------------------------
# Check if student has class now
# --------------------------
def has_class_now(student_id, timetable_dict):
    if student_id not in timetable_dict:
        print(f"❌ No timetable found for student_id: {student_id}")
        return False, None

    ist = pytz.timezone(TIMEZONE)
    now = datetime.now(ist)
    current_day = now.strftime("%A").strip().lower()
    current_time = now.time()

    for entry in timetable_dict[student_id]:
        entry_day = entry["day"].strip().lower()
        
        if isinstance(entry["start"], str):
            start_time = datetime.strptime(entry["start"], "%H:%M:%S").time()
        else:
            start_time = entry["start"]
            
        if isinstance(entry["end"], str):
            end_time = datetime.strptime(entry["end"], "%H:%M:%S").time()
        else:
            end_time = entry["end"]

        if entry_day == current_day and start_time <= current_time <= end_time:
            print(f"✅ Class confirmed: {entry['subject']}")
            return True, entry["subject"]

    print("❌ No matching class found")
    return False, None

# --------------------------
# Simple face recognition using cosine similarity
# --------------------------
def recognize_face(face_embedding):
    if not known_faces:
        return "Unknown", None, 0.0
    
    best_match = "Unknown"
    best_id = None
    best_confidence = 0.0
    
    for i, known_face in enumerate(known_faces):
        similarity = np.dot(face_embedding, known_face) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_face))
        
        if similarity > best_confidence and similarity > THRESHOLD:
            best_confidence = similarity
            best_match = known_names[i]
            best_id = known_ids[i]
    
    return best_match, best_id, best_confidence

# --------------------------
# Simple face embedding extraction
# --------------------------
def extract_simple_embedding(face_image):
    face_resized = cv2.resize(face_image, (160, 160))
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    embedding = face_gray.flatten().astype(np.float32) / 255.0
    return embedding

# --------------------------
# Basic face detection using HOG (if cascade fails)
# --------------------------
def detect_faces_basic(frame):
    # Simple face detection using skin color and contours
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.medianBlur(mask, 5)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    faces = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filter small contours
            x, y, w, h = cv2.boundingRect(contour)
            faces.append((x, y, w, h))
    
    return faces

# --------------------------
# Mark attendance via API
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
# Main attendance system
# --------------------------
def main():
    print("🚀 Starting Simple Face Recognition Attendance System...")
    
    # Load data
    if not load_embeddings():
        print("❌ Failed to load face embeddings. Exiting.")
        return
    
    timetable_dict = load_timetable()
    
    print(f"\n🔧 Configuration:")
    print(f"   - Timezone: {TIMEZONE}")
    print(f"   - Frames required: {FRAMES_REQUIRED}")
    print(f"   - Threshold: {THRESHOLD}")
    print(f"   - Students loaded: {len(known_names)}")
    print(f"   - API Endpoint: {ATTENDANCE_API}")
    print("\n🚀 Starting face recognition...\n")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return

    print("✅ Camera opened successfully")
    print("🔑 Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera")
            break

        # Detect faces
        if face_detector and not face_detector.empty():
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        else:
            # Use basic face detection
            faces = detect_faces_basic(frame)

        if len(faces) == 0:
            cv2.putText(frame, "No faces detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Simple embedding extraction
                embedding = extract_simple_embedding(face_roi)
                
                # Recognize face
                name, student_id, confidence = recognize_face(embedding)
                
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
                        has_class, subject = has_class_now(student_id, timetable_dict)
                        if has_class:
                            print(f"🎯 Attempting to mark attendance for {name} (detections: {current_count}/{FRAMES_REQUIRED})")
                            mark_attendance(name)
                        else:
                            print(f"⏰ No class now for {name}, not marking attendance")
                        detection_counts[name] = 0
                    elif name not in already_marked:
                        if current_count % 5 == 0:
                            print(f"📊 {name}: {current_count}/{FRAMES_REQUIRED} detections")
                else:
                    detection_counts["Unknown"] = 0

        # Display status information
        status_text = f"Marked: {len(already_marked)} | Press 'q' to quit"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Recognition Attendance", frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Quitting...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Cleanup done, program exited.")
    print(f"📊 Summary: Marked attendance for {len(already_marked)} students")
    if already_marked:
        print(f"🎯 Students marked: {', '.join(already_marked)}")

if __name__ == "__main__":
    main()
