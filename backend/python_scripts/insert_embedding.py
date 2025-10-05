import numpy as np
import psycopg2
import os
import sys

# -------------------------
# Get name argument from command line
# -------------------------
if len(sys.argv) < 2:
    print("Name argument missing.")
    exit(1)

name = sys.argv[1]

# -------------------------
# Path to embeddings/<name>.npy
# -------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "..", "embeddings", f"{name}.npy")

if not os.path.exists(file_path):
    print(f"Embedding file not found at: {file_path}")
    exit(1)

# -------------------------
# Load .npy embedding
# -------------------------
embedding = np.load(file_path)
embedding_bytes = embedding.tobytes()  # Store as bytea

# -------------------------
# Connect to PostgreSQL
# -------------------------
try:
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT"),
        database=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASS")
    )
    cursor = conn.cursor()
except Exception as e:
    print(f"PostgreSQL connection error: {e}")
    exit(1)

try:
    # -------------------------
    # Get student_id
    # -------------------------
    cursor.execute("SELECT student_id FROM students WHERE name = %s", (name,))
    result = cursor.fetchone()

    if result is None:
        print(f"No student found with name '{name}'. Please register first.")
        exit(1)

    student_id = result[0]

    # -------------------------
    # Insert embedding
    # -------------------------
    cursor.execute(
        "INSERT INTO face_embeddings (student_id, name, embedding, created_on) VALUES (%s, %s, %s, NOW())",
        (student_id, name, embedding_bytes)
    )
    conn.commit()
    print(f"Embedding inserted for {name} (student_id: {student_id})")

except Exception as e:
    print(f"Error inserting embedding: {e}")
finally:
    cursor.close()
    conn.close()
