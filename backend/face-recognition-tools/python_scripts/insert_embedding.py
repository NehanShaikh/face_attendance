import numpy as np
import psycopg2
import os
import sys

# -------------------------
# Database Configuration - HARDCODED FOR LOCAL USE
# -------------------------
DB_CONFIG = {
    "host": "dpg-d3h73lhr0fns73c5cesg-a.oregon-postgres.render.com",
    "port": 5432,
    "database": "face_db_fym1",
    "user": "face_db_fym1_user",
    "password": "XyD8oZRvbMjx6o5XKmJoFYKjaPWz2uOV"
}

# -------------------------
# Get name argument from command line
# -------------------------
if len(sys.argv) < 2:
    print("❌ Name argument missing.")
    print("Usage: python insert_embedding.py <student_name>")
    exit(1)

name = sys.argv[1]

# -------------------------
# Path to embeddings/<name>.npy
# -------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "embeddings", f"{name}.npy")

if not os.path.exists(file_path):
    print(f"❌ Embedding file not found at: {file_path}")
    print("Please run the face pipeline first to generate embeddings.")
    exit(1)

# -------------------------
# Load .npy embedding
# -------------------------
embedding = np.load(file_path)
embedding_bytes = embedding.tobytes()

print(f"📁 Loaded embedding from: {file_path}")
print(f"📊 Embedding shape: {embedding.shape}")

# -------------------------
# Connect to PostgreSQL
# -------------------------
try:
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("✅ Connected to PostgreSQL database")
except Exception as e:
    print(f"❌ PostgreSQL connection error: {e}")
    exit(1)

try:
    # -------------------------
    # Get student_id
    # -------------------------
    cursor.execute("SELECT student_id FROM students WHERE name = %s", (name,))
    result = cursor.fetchone()

    if result is None:
        print(f"❌ No student found with name '{name}'. Please register student first.")
        exit(1)

    student_id = result[0]
    print(f"👤 Found student: {name} (ID: {student_id})")

    # -------------------------
    # Insert or update embedding
    # -------------------------
    cursor.execute("""
        INSERT INTO face_embeddings (student_id, name, embedding, created_on) 
        VALUES (%s, %s, %s, NOW())
        ON CONFLICT (student_id) 
        DO UPDATE SET 
            embedding = EXCLUDED.embedding,
            created_on = NOW()
    """, (student_id, name, embedding_bytes))
    
    conn.commit()
    print(f"✅ Embedding inserted/updated for {name} (student_id: {student_id})")

except Exception as e:
    print(f"❌ Error inserting embedding: {e}")
    conn.rollback()
finally:
    cursor.close()
    conn.close()
    print("✅ Database connection closed")
