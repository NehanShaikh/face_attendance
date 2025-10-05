from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

# Dummy login endpoint
users = {
    "professor1": {"password": "faculty@1", "role": "faculty"},
    "admin1": {"password": "admin@123", "role": "admin"},
    "nehan": {"password": "12345", "role": "student"}
}

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    user = users.get(username)
    if user and user["password"] == password:
        return jsonify({"role": user["role"]})
    return jsonify({"error": "Invalid credentials"}), 401

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
