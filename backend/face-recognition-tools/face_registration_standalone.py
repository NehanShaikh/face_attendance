import cv2
import os
import numpy as np
import psycopg2
import sys
from datetime import datetime

class FaceRegistration:
    def __init__(self):
        self.db_config = {
            "host": "dpg-d3h73lhr0fns73c5cesg-a.oregon-postgres.render.com",
            "port": 5432,
            "database": "face_db_fym1",
            "user": "face_db_fym1_user",
            "password": "XyD8oZRvbMjx6o5XKmJoFYKjaPWz2uOV"
        }
        
    def capture_images(self, name):
        """Step 1: Capture images"""
        save_dir = f"dataset/{name}"
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return False

        count = 0
        print("📸 Press 'S' to save image, 'Q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cv2.putText(frame, f"Saved: {count}/20 - Press 'S' to capture", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'Q' when done", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Capture Face", frame)
            key = cv2.waitKey(1)
            
            if key == ord('s') or key == ord('S'):
                img_path = f"{save_dir}/{name}_{count:02d}.jpg"
                cv2.imwrite(img_path, frame)
                print(f"✅ Saved: {img_path}")
                count += 1
                
                if count >= 20:
                    break
                    
            elif key == ord('q') or key == ord('Q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        if count > 0:
            print(f"✅ Captured {count} images for {name}")
            return True
        else:
            print("❌ No images captured")
            return False
    
    def crop_faces(self, name):
        """Step 2: Detect and crop faces"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        input_dir = f"dataset/{name}"
        output_dir = f"cropped_faces/{name}"
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(input_dir):
            print(f"❌ Input directory {input_dir} not found")
            return False

        count = 0
        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = img[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (100, 100))
                output_path = os.path.join(output_dir, f"face_{count:03d}.jpg")
                cv2.imwrite(output_path, face_resized)
                count += 1
                print(f"✅ Cropped face: {output_path}")

        print(f"✅ Cropped {count} faces for {name}")
        return count > 0
    
    def generate_512d_embedding(self, face_image):
        """
        Generate EXACT 512-dimensional embedding
        """
        try:
            # Resize to 32x16 = 512 pixels exactly
            face_resized = cv2.resize(face_image, (32, 16))  # 32x16 = 512 pixels
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Flatten to get exactly 512 pixels
            embedding = face_gray.flatten().astype(np.float32) / 255.0
            
            # NO NORMALIZATION - Use raw pixel values (0-1 range)
            print(f"🔧 Embedding: {len(embedding)}D, range: [{np.min(embedding):.3f}, {np.max(embedding):.3f}]")
            return embedding
            
        except Exception as e:
            print(f"❌ Error: {e}")
            # Return valid 512D vector as fallback
            return np.ones(512, dtype=np.float32) * 0.5
    
    def generate_embeddings(self, name):
        """Step 3: Generate embeddings"""
        input_dir = f"cropped_faces/{name}"
        os.makedirs("embeddings", exist_ok=True)

        if not os.path.exists(input_dir):
            print(f"❌ Input directory {input_dir} not found")
            return False

        embeddings = []
        
        print(f"🔧 Generating embeddings for {name}...")
        
        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            embedding = self.generate_512d_embedding(img)
            embeddings.append(embedding)
            print(f"   ✅ {img_name}: {len(embedding)}D")

        if embeddings:
            print(f"✅ Generated {len(embeddings)} embeddings for {name}")
            print(f"📊 Dimension: {len(embeddings[0])}D")
            return embeddings
        else:
            print("❌ No embeddings generated")
            return None
    
    def verify_student_exists(self, name):
        """Check if student exists"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute("SELECT student_id, name FROM students WHERE LOWER(name) = LOWER(%s)", (name,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                student_id, actual_name = result
                print(f"✅ Student found: {actual_name} (ID: {student_id})")
                return student_id, actual_name
            else:
                print(f"❌ Student '{name}' not found")
                return None, None
        except Exception as e:
            print(f"❌ Database error: {e}")
            return None, None
    
    def insert_embedding(self, name, embeddings):
        """Step 4: Insert embedding into database"""
        if not embeddings:
            print("❌ No embeddings to insert")
            return False

        # Use average embedding
        avg_embedding = np.mean(embeddings, axis=0)
        
        print(f"📊 Average embedding: {len(avg_embedding)}D")
        
        # Verify it's exactly 512D
        if len(avg_embedding) != 512:
            print(f"❌ Wrong dimension: {len(avg_embedding)} (expected: 512)")
            return False

        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Verify student exists
            student_id, actual_name = self.verify_student_exists(name)
            if not student_id:
                return False
            
            # Convert to bytes (512 floats * 4 bytes = 2048 bytes)
            emb_bytes = avg_embedding.tobytes()
            emb_hex = emb_bytes.hex()
            
            print(f"🔍 Storage details:")
            print(f"   - Floats: {len(avg_embedding)}")
            print(f"   - Bytes: {len(emb_bytes)}")
            print(f"   - Hex chars: {len(emb_hex)}")
            
            # DELETE EXISTING FIRST
            cursor.execute("DELETE FROM face_embeddings WHERE student_id = %s", (student_id,))
            print(f"🧹 Deleted existing embedding")
            
            # Insert new embedding
            cursor.execute("""
                INSERT INTO face_embeddings (student_id, name, embedding, created_on) 
                VALUES (%s, %s, %s, %s)
            """, (student_id, actual_name, emb_hex, datetime.now()))
            
            conn.commit()
            print(f"✅ Inserted embedding for '{actual_name}'")
            
            # Verify storage
            cursor.execute("SELECT LENGTH(embedding) FROM face_embeddings WHERE student_id = %s", (student_id,))
            hex_length = cursor.fetchone()[0]
            
            expected_hex_length = 512 * 4 * 2  # 512 floats * 4 bytes * 2 hex chars
            print(f"🔍 Verification:")
            print(f"   - Expected hex length: {expected_hex_length}")
            print(f"   - Actual hex length: {hex_length}")
            
            if hex_length == expected_hex_length:
                print("   ✅ STORAGE CORRECT - 512D embedding stored properly!")
            else:
                print(f"   ❌ STORAGE WRONG - Got {hex_length}, expected {expected_hex_length}")
                return False
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
    
    def cleanup_old_data(self, name):
        """Clean up old data"""
        import shutil
        
        folders = [
            f"dataset/{name}",
            f"cropped_faces/{name}"
        ]
        
        for folder in folders:
            if os.path.exists(folder):
                if os.path.isdir(folder):
                    shutil.rmtree(folder)
    
    def register_student(self, name):
        """Complete registration pipeline"""
        print(f"🚀 Starting Face Registration for: {name}")
        print("=" * 50)
        
        # Clean up first
        self.cleanup_old_data(name)
        
        # Step 1: Capture images
        print(f"\n▶️ Capturing images...")
        if not self.capture_images(name):
            return False
        print(f"✅ Image capture completed")
        
        # Step 2: Crop faces
        print(f"\n▶️ Cropping faces...")
        if not self.crop_faces(name):
            return False
        print(f"✅ Face cropping completed")
        
        # Step 3: Generate embeddings
        print(f"\n▶️ Generating embeddings...")
        embeddings = self.generate_embeddings(name)
        if embeddings is None:
            return False
        print(f"✅ Embedding generation completed")
        
        # Step 4: Save to database
        print(f"\n▶️ Saving to database...")
        if not self.insert_embedding(name, embeddings):
            return False
        print(f"✅ Database save completed")
        
        print(f"\n🎉 Face registration completed for {name}!")
        print("=" * 50)
        return True

def main():
    print("🎯 Face Registration Tool")
    print("=" * 30)
    
    name = input("Enter student name: ").strip()
    
    if not name:
        print("❌ Name cannot be empty")
        return
    
    registrar = FaceRegistration()
    success = registrar.register_student(name)
    
    if success:
        print(f"\n✅ SUCCESS: {name} registered with 512D embeddings!")
        print("Run your recognition system to test.")
    else:
        print(f"\n❌ FAILED: Registration failed")

if __name__ == "__main__":
    main()
