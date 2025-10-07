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
# Initialize Face Detector
# --------------------------
def load_face_detector():
    print("🚀 Loading face detection model...")
    try:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if not detector.empty():
            print("✅ Face detector loaded successfully")
            return detector
    except:
        print("❌ Using basic face detection")
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
                    print(f"✅ Loaded embedding for {name} (dimension: {len(emb)})")
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
# SIMPLE FACE RECOGNITION - FIXED VERSION
# --------------------------
def recognize_face_simple(face_embedding):
    if not known_faces:
        return "Unknown", None, 0.0
    
    best_match = "Unknown"
    best_id = None
    best_confidence = 0.0
    
    # Get target dimension from first known face
    target_dim = len(known_faces[0])
    
    # Resize current face embedding to match database dimension
    if len(face_embedding) != target_dim:
        # Simple resize: take first target_dim elements or pad with zeros
        if len(face_embedding) > target_dim:
            face_embedding_resized = face_embedding[:target_dim]
        else:
            face_embedding_resized = np.pad(face_embedding, (0, target_dim - len(face_embedding)))
    else:
        face_embedding_resized = face_embedding
    
    for i, known_face in enumerate(known_faces):
        # Ensure both vectors have same dimension
        if len(known_face) != len(face_embedding_resized):
            continue
            
        # Calculate cosine similarity
        dot_product = np.dot(face_embedding_resized, known_face)
        norm_a = np.linalg.norm(face_embedding_resized)
        norm_b = np.linalg.norm(known_face)
        
        if norm_a == 0 or norm_b == 0:
            continue
            
        similarity = dot_product / (norm_a * norm_b)
        
        if similarity > best_confidence and similarity > THRESHOLD:
            best_confidence = similarity
            best_match = known_names[i]
            best_id = known_ids[i]
    
    return best_match, best_id, best_confidence

# --------------------------
# SIMPLE EMBEDDING EXTRACTION - FIXED
# --------------------------
def extract_simple_embedding(face_image):
    # Resize face to match FaceNet-like dimensions but simpler
    face_resized = cv2.resize(face_image, (64, 64))  # Smaller size
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply some basic feature extraction
    # 1. Histogram of Oriented Gradients (simplified)
    gx = cv2.Sobel(face_gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(face_gray, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx, gy)
    
    # 2. Create a simple feature vector
    hog_features = []
    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            cell_magnitude = magnitude[i:i+8, j:j+8]
            cell_angle = angle[i:i+8, j:j+8]
            hist = np.histogram(cell_angle, bins=8, range=(0, 2*np.pi), weights=cell_magnitude)[0]
            hog_features.extend(hist)
    
    # 3. Combine with flattened image data
    flattened = face_gray.flatten().astype(np.float32) / 255.0
    
    # 4. Create final embedding (target ~512 dimensions like FaceNet)
    embedding = np.concatenate([flattened[:256], hog_features[:256]])
    
    return embedding

# --------------------------
# DEMO EMBEDDING (if database embeddings are not compatible)
# --------------------------
def extract_demo_embedding(face_image):
    # Create a consistent 512-dimensional embedding for demo
    face_resized = cv2.resize(face_image, (32, 32))
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    # Simple features that create 512-dim vector
    flattened = face_gray.flatten().astype(np.float32) / 255.0
    
    # If we need more dimensions, repeat or create patterns
    if len(flattened) < 512:
        # Repeat pattern to reach 512 dimensions
        repeated = np.tile(flattened, 512 // len(flattened) + 1)
        embedding = repeated[:512]
    else:
        embedding = flattened[:512]
    
    return embedding

# --------------------------
# Basic face detection fallback
# --------------------------
def detect_faces_basic(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.medianBlur(mask, 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    faces = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
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
    print("🚀 Starting Face Recognition Attendance System...")
    
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
            faces = detect_faces_basic(frame)

        if len(faces) == 0:
            cv2.putText(frame, "No faces detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Choose embedding method based on database embeddings
                if known_faces and len(known_faces[0]) == 512:
                    # Use demo embedding to match 512-dim database embeddings
                    embedding = extract_demo_embedding(face_roi)
                else:
                    # Use simple embedding
                    embedding = extract_simple_embedding(face_roi)
                
                print(f"🔍 Extracted embedding dimension: {len(embedding)}")
                
                # Recognize face
                name, student_id, confidence = recognize_face_simple(embedding)
                
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
