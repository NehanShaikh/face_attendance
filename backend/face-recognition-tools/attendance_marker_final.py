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
THRESHOLD = 0.6  # Increased threshold for better discrimination
FRAMES_REQUIRED = 5
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
    except Exception as e:
        print(f"❌ Error loading face detector: {e}")
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
        return conn
    except Exception as e:
        print(f"❌ PostgreSQL connection error: {e}")
        return None

# --------------------------
# Load saved embeddings from database - IMPROVED VERSION
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
            ORDER BY s.student_id
        """)

        loaded_count = 0
        problematic_embeddings = []
        
        for student_id, name, emb_data in cursor.fetchall():
            if emb_data:
                try:
                    # Handle different data types from database
                    if isinstance(emb_data, memoryview):
                        emb_data = emb_data.tobytes()
                    elif isinstance(emb_data, str):
                        if emb_data.startswith("\\x"):
                            emb_data = bytes.fromhex(emb_data[2:])
                        else:
                            emb_data = emb_data.encode('latin-1')
                    
                    # Parse the embedding
                    emb = np.frombuffer(emb_data, dtype=np.float32)
                    
                    # Skip empty embeddings
                    if len(emb) == 0:
                        print(f"⚠️ Empty embedding for {name}")
                        continue
                    
                    # Check for problematic embeddings (all zeros, all same value, etc.)
                    if np.all(emb == 0) or np.std(emb) < 0.001:
                        print(f"❌ Problematic embedding for {name} - all zeros or low variance")
                        problematic_embeddings.append(name)
                        continue
                    
                    # Normalize the embedding
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    else:
                        print(f"❌ Zero norm embedding for {name}")
                        continue
                    
                    known_faces.append(emb)
                    known_names.append(name)
                    known_ids.append(student_id)
                    loaded_count += 1
                    print(f"✅ Loaded {name} - Dim: {len(emb)}, Norm: {np.linalg.norm(emb):.4f}")
                    
                except Exception as e:
                    print(f"❌ Failed to parse embedding for {name}: {e}")
            else:
                print(f"⚠️ No embedding found for {name}")

        cursor.close()
        conn.close()
        
        print(f"\n📊 DATABASE SUMMARY:")
        print(f"   - Successfully loaded: {loaded_count} embeddings")
        print(f"   - Problematic: {len(problematic_embeddings)} embeddings")
        
        if problematic_embeddings:
            print(f"   ❌ Problematic embeddings: {problematic_embeddings}")
            print("   💡 Re-register these students using the standalone registration script")
        
        if known_faces:
            dimensions = [len(emb) for emb in known_faces]
            unique_dims = set(dimensions)
            print(f"   - Unique dimensions: {unique_dims}")
            
            if len(unique_dims) > 1:
                print("   ❌ WARNING: Dimension mismatch detected!")
        
        return loaded_count > 0
        
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
            return True, entry["subject"]

    return False, None

# --------------------------
# ENHANCED EMBEDDING EXTRACTION
# --------------------------
def extract_enhanced_embedding(face_image):
    """
    Enhanced embedding extraction with multiple features
    Consistent 512 dimensions with better discrimination
    """
    try:
        # Resize to standard size
        face_resized = cv2.resize(face_image, (64, 64))
        
        features = []
        
        # 1. Grayscale features
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        gray_normalized = face_gray.flatten().astype(np.float32) / 255.0
        features.extend(gray_normalized[:150])  # Take first 150 pixels
        
        # 2. Color histograms (RGB)
        for i in range(3):
            hist = cv2.calcHist([face_resized], [i], None, [32], [0, 256])
            hist_normalized = hist.flatten() / np.sum(hist) if np.sum(hist) > 0 else hist.flatten()
            features.extend(hist_normalized)
        
        # 3. Edge features
        sobelx = cv2.Sobel(face_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(face_gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        features.extend([
            np.mean(edge_magnitude),
            np.std(edge_magnitude),
            np.max(edge_magnitude)
        ])
        
        # 4. Texture features
        blur = cv2.GaussianBlur(face_gray, (3, 3), 0)
        features.extend([
            np.mean(blur),
            np.std(blur)
        ])
        
        # Convert to numpy array
        embedding = np.array(features, dtype=np.float32)
        
        # Ensure exactly 512 dimensions
        target_dim = 512
        if len(embedding) < target_dim:
            # Pad with small random values (better than zeros)
            padding = np.random.normal(0, 0.01, target_dim - len(embedding)).astype(np.float32)
            embedding = np.concatenate([embedding, padding])
        else:
            embedding = embedding[:target_dim]
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
        
    except Exception as e:
        print(f"❌ Error in embedding extraction: {e}")
        # Return random embedding as fallback
        return np.random.normal(0, 0.1, 512).astype(np.float32)

# --------------------------
# ROBUST FACE RECOGNITION
# --------------------------
def recognize_face_robust(face_embedding):
    """Robust face recognition with multiple validation checks"""
    if not known_faces:
        return "Unknown", None, 0.0
    
    best_match = "Unknown"
    best_id = None
    best_confidence = 0.0
    
    # Validate current embedding
    if np.all(face_embedding == 0) or np.std(face_embedding) < 0.001:
        print("❌ Invalid current embedding - all zeros or low variance")
        return "Unknown", None, 0.0
    
    current_norm = np.linalg.norm(face_embedding)
    if abs(current_norm - 1.0) > 0.1:  # Should be close to 1 after normalization
        print(f"⚠️ Current embedding not properly normalized: {current_norm:.4f}")
        # Re-normalize
        face_embedding = face_embedding / current_norm
    
    print(f"\n🔍 Recognizing face...")
    print(f"   - Current embedding: {len(face_embedding)}D, norm: {np.linalg.norm(face_embedding):.4f}")
    
    for i, known_face in enumerate(known_faces):
        student_name = known_names[i]
        known_dim = len(known_face)
        current_dim = len(face_embedding)
        
        # Skip if dimensions are too different
        if known_dim != current_dim:
            continue
            
        # Validate known embedding
        known_norm = np.linalg.norm(known_face)
        if abs(known_norm - 1.0) > 0.1:
            print(f"⚠️ {student_name} embedding not normalized: {known_norm:.4f}")
            continue
        
        # Calculate cosine similarity
        similarity = np.dot(face_embedding, known_face)
        
        # Clamp to valid range
        similarity = np.clip(similarity, -1.0, 1.0)
        
        print(f"   🔍 {student_name}: {similarity:.4f}")
        
        # Update best match if better and above threshold
        if similarity > best_confidence and similarity > THRESHOLD:
            best_confidence = similarity
            best_match = student_name
            best_id = known_ids[i]
    
    # Final decision with confidence check
    if best_match != "Unknown":
        print(f"   ✅ MATCH: {best_match} (confidence: {best_confidence:.4f})")
        
        # Additional confidence check
        if best_confidence < 0.7:  # Moderate confidence threshold
            print(f"   ⚠️ Low confidence match: {best_confidence:.4f}")
    else:
        if best_confidence > 0:
            print(f"   ❌ No confident match (best: {best_confidence:.4f}, threshold: {THRESHOLD})")
        else:
            print("   ❌ No match found")
    
    return best_match, best_id, best_confidence

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
            print(f"📤 Marking attendance for: {name}")
            r = requests.post(ATTENDANCE_API, json={"name": name}, timeout=10)
            if r.ok:
                already_marked.add(name)
                print(f"✅ Attendance marked: {name}")
            else:
                print(f"❌ API failed ({r.status_code})")
        except Exception as e:
            print(f"❌ Request error: {e}")

# --------------------------
# VALIDATE EMBEDDINGS
# --------------------------
def validate_embeddings():
    """Validate all loaded embeddings"""
    print("\n🔍 VALIDATING EMBEDDINGS...")
    valid_count = 0
    
    for i, (name, embedding) in enumerate(zip(known_names, known_faces)):
        norm = np.linalg.norm(embedding)
        std = np.std(embedding)
        
        status = "✅" if (0.9 <= norm <= 1.1 and std > 0.01) else "❌"
        print(f"   {status} {name}: norm={norm:.4f}, std={std:.4f}, dim={len(embedding)}")
        
        if 0.9 <= norm <= 1.1 and std > 0.01:
            valid_count += 1
    
    print(f"📊 Valid embeddings: {valid_count}/{len(known_faces)}")
    return valid_count == len(known_faces)

# --------------------------
# Main attendance system
# --------------------------
def main():
    print("🚀 Starting Face Recognition Attendance System...")
    print("🔧 ROBUST VERSION - Proper Multi-Student Recognition")
    
    # Load data
    if not load_embeddings():
        print("❌ Failed to load face embeddings. Exiting.")
        return
    
    # Validate embeddings
    if not validate_embeddings():
        print("❌ Some embeddings are invalid. Please re-register problematic students.")
        print("💡 Use the standalone registration script to fix this.")
    
    timetable_dict = load_timetable()
    
    print(f"\n🔧 Configuration:")
    print(f"   - Students loaded: {len(known_names)}")
    print(f"   - Threshold: {THRESHOLD}")
    print(f"   - Frames required: {FRAMES_REQUIRED}")
    print("\n🎯 Ready for recognition...\n")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return

    print("✅ Camera opened successfully")
    print("🔑 Press 'q' to quit\n")

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera")
            break

        frame_count += 1
        
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
                # Extract face region with padding
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_roi = frame[y1:y2, x1:x2]
                
                # Use enhanced embedding extraction
                embedding = extract_enhanced_embedding(face_roi)
                
                # Recognize face
                name, student_id, confidence = recognize_face_robust(embedding)
                
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
                            print(f"🎯 Marking attendance for {name} ({current_count} detections)")
                            mark_attendance(name)
                        else:
                            print(f"⏰ No class for {name}")
                        detection_counts[name] = 0

        # Display status information
        status_text = f"Marked: {len(already_marked)} | Students: {len(known_names)} | Frame: {frame_count}"
        cv2.putText(frame, status_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Recognition - Multi-Student", frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n🛑 Quitting...")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   - Marked attendance for: {len(already_marked)} students")
    if already_marked:
        print(f"   - Students: {', '.join(already_marked)}")
    print("✅ Program completed successfully")

if __name__ == "__main__":
    main()
