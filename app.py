from flask import Flask, request, jsonify
from knn_engine import get_recommendations

app = Flask(__name__)

@app.route("/match/calculate", methods=["POST"])
def match():
    data = request.get_json()
    student_id = data.get("studentId")
    if student_id is None:
        return jsonify({"error": "studentId is required"}), 400

    try:
        recommendations = get_recommendations(student_id)
        return jsonify(recommendations)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
