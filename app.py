# app.py
import mysql
from flask import Flask, jsonify, request
from flask_cors import CORS
from model import generate_response

app = Flask(__name__)
CORS(app)  # Allow frontend to access the API

# Database connection config
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "",  # use your actual password if you set one
    "database": "presentation"  # change to your DB name
}


@app.route('/training-data', methods=['GET'])
def get_training_data():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)  # dictionary=True returns rows as dicts
        cursor.execute("SELECT prompt, response FROM training_data")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()
        mode = data.get("mode", "pretrained").strip().lower()

        if not message:
            return jsonify({"error": "Empty message"}), 400

        if mode not in ["pretrained", "fine-tuned"]:
            return jsonify({"error": "Invalid mode. Use 'pretrained' or 'fine-tuned'."}), 400

        response = generate_response(message, mode=mode)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
