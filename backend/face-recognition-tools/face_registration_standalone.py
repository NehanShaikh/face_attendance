import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import os
import numpy as np
import psycopg2
import pickle
import sys

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
        """Step 1: Capture images manually"""
        save_dir = f"dataset/{name}"
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return False

        count = 0
        messagebox.showinfo("Instructions", "Press 'S' to save image\nPress 'Q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            cv2.putText(frame, f"Saved: {count}/20 - Press 'S' to capture", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'Q' when done", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Capture Face - Press S to Save, Q to Quit", frame)
            key = cv2.waitKey(1)
            
            if key == ord('s') or key == ord('S'):
                img_path = f"{save_dir}/{name}_{count}.jpg"
                cv2.imwrite(img_path, frame)
                print(f"✅ Saved: {img_path}")
                count += 1
                
                if count >= 20:  # Stop after 20 images
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
                output_path = os.path.join(output_dir, f"face_{count}.jpg")
                cv2.imwrite(output_path, face)
                count += 1

        print(f"✅ Cropped {count} faces for {name}")
        return count > 0
    
    def generate_embeddings(self, name):
        """Step 3: Generate face embeddings"""
        input_dir = f"cropped_faces/{name}"
        output_file = f"embeddings/{name}_embeddings.pkl"
        os.makedirs("embeddings", exist_ok=True)

        if not os.path.exists(input_dir):
            print(f"❌ Input directory {input_dir} not found")
            return False

        embeddings = []
        for img_name in os.listdir(input_dir):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Create 512-dimensional embedding
            face_resized = cv2.resize(img, (64, 64))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            embedding = face_gray.flatten().astype(np.float32) / 255.0
            
            # Ensure 512 dimensions
            if len(embedding) < 512:
                embedding = np.pad(embedding, (0, 512 - len(embedding)))
            else:
                embedding = embedding[:512]
            
            embeddings.append(embedding)

        # Save embeddings
        with open(output_file, 'wb') as f:
            pickle.dump(embeddings, f)

        print(f"✅ Generated {len(embeddings)} embeddings for {name}")
        return len(embeddings) > 0
    
    def insert_embedding(self, name):
        """Step 4: Insert embedding into database - FIXED TO INCLUDE NAME"""
        emb_file = f"embeddings/{name}_embeddings.pkl"

        if not os.path.exists(emb_file):
            print(f"❌ Embeddings file {emb_file} not found")
            return False

        with open(emb_file, 'rb') as f:
            embeddings = pickle.load(f)

        if not embeddings:
            print("❌ No embeddings found")
            return False

        # Use average embedding
        avg_embedding = np.mean(embeddings, axis=0)

        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            print(f"🔍 Looking for student with name: '{name}'")
            
            # First, find the student by name (case-insensitive search)
            cursor.execute("SELECT student_id, name FROM students WHERE LOWER(name) = LOWER(%s)", (name,))
            result = cursor.fetchone()
            
            if not result:
                print(f"❌ No student found with name: '{name}'")
                
                # Show available students for debugging
                print("💡 Available students in database:")
                cursor.execute("SELECT student_id, name FROM students ORDER BY student_id")
                available_students = cursor.fetchall()
                if available_students:
                    for sid, sname in available_students:
                        print(f"   - ID: {sid}, Name: '{sname}'")
                else:
                    print("   - No students found in database")
                
                return False
            
            student_id, actual_name = result
            print(f"✅ Found student - ID: {student_id}, Name: '{actual_name}'")
            
            # Convert embedding to bytes then to hex string for TEXT column
            emb_bytes = avg_embedding.tobytes()
            emb_hex = emb_bytes.hex()  # Convert to hex string for TEXT column
            print(f"🔍 Embedding size: {len(emb_bytes)} bytes, Hex length: {len(emb_hex)}")
            
            # Check if embedding already exists for this student
            cursor.execute("SELECT embedding_id FROM face_embeddings WHERE student_id = %s", (student_id,))
            existing_embedding = cursor.fetchone()
            
            if existing_embedding:
                # Update existing embedding WITH NAME
                cursor.execute("""
                    UPDATE face_embeddings 
                    SET embedding = %s, name = %s
                    WHERE student_id = %s
                """, (emb_hex, actual_name, student_id))
                print(f"🔄 Updated existing face embedding for '{actual_name}'")
            else:
                # Insert new embedding WITH NAME
                cursor.execute("""
                    INSERT INTO face_embeddings (student_id, name, embedding) 
                    VALUES (%s, %s, %s)
                """, (student_id, actual_name, emb_hex))
                print(f"✅ Inserted new face embedding for '{actual_name}'")
            
            # Verify the insertion
            cursor.execute("""
                SELECT embedding_id, student_id, name, LENGTH(embedding) as embedding_length
                FROM face_embeddings 
                WHERE student_id = %s
            """, (student_id,))
            verification = cursor.fetchone()
            
            conn.commit()
            
            if verification:
                emb_id, verified_id, verified_name, embedding_length = verification
                print(f"🎉 VERIFICATION SUCCESS!")
                print(f"   - Embedding ID: {emb_id}")
                print(f"   - Student ID: {verified_id}")
                print(f"   - Student Name: '{verified_name}'")
                print(f"   - Embedding Length: {embedding_length} characters")
                print(f"   - Face registration completed for '{verified_name}'")
            else:
                print("❌ VERIFICATION FAILED - No embedding found after insertion")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Database error: {e}")
            import traceback
            traceback.print_exc()
            if 'conn' in locals():
                conn.rollback()
            return False
        finally:
            if 'conn' in locals():
                cursor.close()
                conn.close()
    
    def register_student(self, name):
        """Complete registration pipeline"""
        print(f"🚀 Starting face registration for: {name}")
        
        steps = [
            ("📸 Capturing images", self.capture_images),
            ("✂️ Cropping faces", self.crop_faces),
            ("🔮 Generating embeddings", self.generate_embeddings),
            ("💾 Saving to database", self.insert_embedding)
        ]
        
        for step_name, step_func in steps:
            print(f"\n▶️ {step_name}...")
            if not step_func(name):
                print(f"❌ Failed at: {step_name}")
                return False
            print(f"✅ {step_name} completed")
        
        print(f"\n🎉 Face registration completed for {name}!")
        return True

def main():
    root = tk.Tk()
    root.withdraw()
    
    name = simpledialog.askstring("Face Registration", "Enter student name:")
    
    if not name:
        messagebox.showwarning("Cancelled", "Registration cancelled.")
        return
    
    registrar = FaceRegistration()
    success = registrar.register_student(name)
    
    if success:
        messagebox.showinfo("Success", f"Face registration completed for {name}!")
    else:
        messagebox.showerror("Error", f"Registration failed for {name}. Check console for details.")
    
    root.destroy()

if __name__ == "__main__":
    main()
