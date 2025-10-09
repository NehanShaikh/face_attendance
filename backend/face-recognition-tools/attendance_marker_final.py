import cv2
import numpy as np
import requests
import psycopg2
import os
import sys
from datetime import datetime
import pytz

# --------------------------
# Config
# --------------------------
THRESHOLD = 0.7  # Increased threshold for better discrimination
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
# Global variables
# --------------------------
already_marked = set()
detection_counts = {}
known_faces = []
known_names = []
known_ids = []
EMBEDDING_DIM = 512  # Default, will be updated from database

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
# Load embeddings from database - ADAPTS TO ANY DIMENSION
# --------------------------
def load_embeddings():
    global known_faces, known_names, known_ids, EMBEDDING_DIM
    
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

        loaded_count = 0
        dimensions_found = set()
        
        for student_id, name, emb_data in cursor.fetchall():
            if emb_data:
                try:
                    print(f"🔍 Processing {name}")
                    
                    # Handle the string data - it's already hex encoded
                    if isinstance(emb_data, str):
                        # Direct hex string (4096 chars = 2048 bytes = 512 floats)
                        if len(emb_data) == 4096:
                            # This is 512D embedding (512 floats * 4 bytes * 2 hex chars = 4096)
                            emb_bytes = bytes.fromhex(emb_data)
                            EMBEDDING_DIM = 512
                        elif len(emb_data) == 8192:
                            # This is 1024D embedding (1024 floats * 4 bytes * 2 hex chars = 8192)
                            emb_bytes = bytes.fromhex(emb_data)
                            EMBEDDING_DIM = 1024
                        else:
                            # Unknown format, try to parse as hex
                            emb_bytes = bytes.fromhex(emb_data)
                            EMBEDDING_DIM = len(emb_bytes) // 4  # 4 bytes per float
                    
                    # Parse to numpy array
                    emb = np.frombuffer(emb_bytes, dtype=np.float32)
                    
                    print(f"   - Dimension: {len(emb)}D")
                    dimensions_found.add(len(emb))
                    
                    # Check if embedding is valid
                    if np.all(emb == 0) or np.std(emb) < 0.001:
                        print(f"   ❌ Invalid embedding (all zeros or low variance)")
                        continue
                    
                    # Normalize the embedding for better cosine similarity
                    emb_norm = emb / np.linalg.norm(emb)
                    
                    # Store the normalized embedding
                    known_faces.append(emb_norm)
                    known_names.append(name)
                    known_ids.append(student_id)
                    loaded_count += 1
                    
                    print(f"   ✅ Loaded successfully")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            else:
                print(f"⚠️ No embedding data for {name}")

        cursor.close()
        conn.close()
        
        # Set the global embedding dimension based on what we found
        if dimensions_found:
            EMBEDDING_DIM = list(dimensions_found)[0]
            print(f"🔧 Using embedding dimension: {EMBEDDING_DIM}D")
        
        print(f"\n📊 DATABASE SUMMARY:")
        print(f"   - Loaded: {loaded_count} embeddings")
        print(f"   - Dimension: {EMBEDDING_DIM}D")
        print(f"   - Students: {known_names}")
        
        return loaded_count > 0
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False

# --------------------------
# Extract embedding - ADAPTS TO DATABASE DIMENSION
# --------------------------
def extract_embedding(face_image):
    """
    Extract embedding that matches database dimension
    """
    try:
        if EMBEDDING_DIM == 512:
            # 512D: 32x16 = 512 pixels
            face_resized = cv2.resize(face_image, (32, 16))
        elif EMBEDDING_DIM == 1024:
            # 1024D: 32x32 = 1024 pixels
            face_resized = cv2.resize(face_image, (32, 32))
        else:
            # Fallback: resize to get exact dimension
            side = int(np.sqrt(EMBEDDING_DIM))
            face_resized = cv2.resize(face_image, (side, side))
        
        face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Flatten to get exact dimension
        embedding = face_gray.flatten().astype(np.float32) / 255.0
        
        # Ensure exact dimension
        if len(embedding) > EMBEDDING_DIM:
            embedding = embedding[:EMBEDDING_DIM]
        elif len(embedding) < EMBEDDING_DIM:
            embedding = np.pad(embedding, (0, EMBEDDING_DIM - len(embedding)))
        
        # Normalize the embedding for better cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        print(f"🔧 Extracted: {len(embedding)}D, Norm: {np.linalg.norm(embedding):.3f}")
        return embedding
        
    except Exception as e:
        print(f"❌ Error in embedding extraction: {e}")
        return np.ones(EMBEDDING_DIM, dtype=np.float32) * 0.5

# --------------------------
# IMPROVED Face Recognition with Better Unknown Handling
# --------------------------
def recognize_face(face_embedding):
    """Improved face recognition with better unknown detection"""
    if not known_faces:
        return "Unknown", None, 0.0
    
    best_match = "Unknown"
    best_id = None
    best_similarity = 0.0
    
    print(f"🎯 Recognizing: {len(face_embedding)}D vs {len(known_faces[0])}D")
    
    for i, known_face in enumerate(known_faces):
        # Handle dimension mismatch by using minimum dimension
        current_dim = len(face_embedding)
        known_dim = len(known_face)
        
        if current_dim != known_dim:
            min_dim = min(current_dim, known_dim)
            face_emb_resized = face_embedding[:min_dim]
            known_face_resized = known_face[:min_dim]
            print(f"   ⚠️ Dimension mismatch: using {min_dim}D")
        else:
            face_emb_resized = face_embedding
            known_face_resized = known_face
        
        # Calculate cosine similarity (both vectors are normalized)
        similarity = np.dot(face_emb_resized, known_face_resized)
        
        # Debug info
        print(f"   {known_names[i]}: {similarity:.3f}")
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = known_names[i]
            best_id = known_ids[i]
    
    # CRITICAL: Only return a match if it exceeds threshold
    if best_similarity < THRESHOLD:
        print(f"❌ No good match (best: {best_similarity:.3f} < {THRESHOLD})")
        return "Unknown", None, best_similarity
    else:
        print(f"✅ Recognized: {best_match} (similarity: {best_similarity:.3f})")
        return best_match, best_id, best_similarity

# --------------------------
# Face Detection
# --------------------------
def detect_faces(frame):
    """Detect faces using Haar Cascade"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    return faces

# --------------------------
# Mark attendance
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
# Main system
# --------------------------
def main():
    print("🚀 Starting Face Recognition System")
    print("🔧 IMPROVED VERSION - Better Unknown Face Detection")
    
    # Load embeddings
    if not load_embeddings():
        print("❌ Failed to load embeddings")
        return
    
    print(f"\n🔧 Configuration:")
    print(f"   - Students: {len(known_names)}")
    print(f"   - Dimension: {EMBEDDING_DIM}D")
    print(f"   - Threshold: {THRESHOLD}")
    print(f"   - Frames Required: {FRAMES_REQUIRED}")
    print("\n🎯 Ready for recognition...\n")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Camera opened")
    print("🔑 Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detect_faces(frame)

        if len(faces) == 0:
            cv2.putText(frame, "No faces detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                # Extract face with padding
                padding = 20
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(frame.shape[1], x + w + padding)
                y2 = min(frame.shape[0], y + h + padding)
                
                face_roi = frame[y1:y2, x1:x2]
                
                # Extract embedding
                embedding = extract_embedding(face_roi)
                
                # Recognize
                name, student_id, confidence = recognize_face(embedding)
                
                # Draw results with appropriate colors
                if name == "Unknown":
                    color = (0, 0, 255)  # Red for unknown
                    label = f"Unknown ({confidence:.2f})"
                else:
                    color = (0, 255, 0)  # Green for known
                    label = f"{name} ({confidence:.2f})"
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Attendance logic (only for known faces)
                if name != "Unknown":
                    detection_counts[name] = detection_counts.get(name, 0) + 1
                    current_count = detection_counts[name]
                    
                    # Show detection progress
                    progress = f"Detected: {current_count}/{FRAMES_REQUIRED}"
                    cv2.putText(frame, progress, (x, y + h + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    if current_count >= FRAMES_REQUIRED and name not in already_marked:
                        mark_attendance(name)
                        detection_counts[name] = 0  # Reset counter after marking

        # Status
        status = f"Marked: {len(already_marked)}/{len(known_names)} | Dim: {EMBEDDING_DIM}D | Thresh: {THRESHOLD}"
        cv2.putText(frame, status, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Face Recognition - Improved", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Summary: Marked {len(already_marked)} students")
    if already_marked:
        print(f"🎯 Students: {', '.join(already_marked)}")

if __name__ == "__main__":
    main()
