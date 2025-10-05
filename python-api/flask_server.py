from flask import Flask, request, jsonify
import subprocess
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # To allow requests from the frontend

@app.route('/capture-faces', methods=['POST'])
def capture_faces():
    subprocess.run(["python3", "1_capture_images.py"])
    return jsonify({"message": "Face capturing done."})

@app.route('/crop-faces', methods=['POST'])
def crop_faces():
    subprocess.run(["python3", "2_crop_faces.py"])
    return jsonify({"message": "Face cropping done."})

@app.route('/generate-embeddings', methods=['POST'])
def generate_embeddings():
    subprocess.run(["python3", "3_generate_embeddings.py"])
    return jsonify({"message": "Embeddings generated."})

@app.route('/start-attendance', methods=['POST'])
def start_attendance():
    subprocess.run(["python3", "4_face_recognition.py"])
    return jsonify({"message": "Attendance started."})

if __name__ == '__main__':
    app.run(port=5000)
