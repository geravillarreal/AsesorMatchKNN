from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException, NotFound

from knn_engine import get_recommendations
from middleware import log_request
from validation import validate_match_request

app = Flask(__name__)

app.before_request(log_request)


@app.errorhandler(HTTPException)
def handle_http_error(error):
    return jsonify({"error": error.description}), error.code


@app.errorhandler(Exception)
def handle_generic_error(error):
    return (
        jsonify({"error": "Internal server error", "detail": str(error)}),
        500,
    )

@app.route("/match/calculate", methods=["POST"])
def match():
    data = request.get_json()
    student_id = validate_match_request(data)
    try:
        recommendations = get_recommendations(student_id)
    except ValueError as e:
        raise NotFound(str(e))
    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
