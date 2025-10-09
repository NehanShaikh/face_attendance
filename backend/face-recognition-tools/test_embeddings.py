# test_embeddings.py
import pickle
import numpy as np

def test_embeddings():
    """Test if embeddings are properly generated"""
    try:
        with open('embeddings/ns_embeddings.npy', 'rb') as f:
            ns_embeddings = pickle.load(f)
        print(f"✅ ns embeddings: {len(ns_embeddings)} embeddings, dimension: {len(ns_embeddings[0])}")
    except:
        print("❌ No ns embeddings found")
    
    # Test other students
    students = ['nehan', 'hari', 'shreyas']  # Add your actual student names
    for student in students:
        try:
            with open(f'embeddings/{student}_embeddings.pkl', 'rb') as f:
                emb = pickle.load(f)
            print(f"✅ {student} embeddings: {len(emb)} embeddings, dimension: {len(emb[0])}")
        except:
            print(f"❌ No {student} embeddings found")

if __name__ == "__main__":
    test_embeddings()
