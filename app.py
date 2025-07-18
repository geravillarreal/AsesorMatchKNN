from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException, NotFound, BadRequest

from knn_engine import get_recommendations
from middleware import log_request
from validation import validate_match_request, Recommendation
from auth import auth_required, create_token

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


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    user_id = data.get("userId")
    if user_id is None:
        raise BadRequest("userId is required")
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        raise BadRequest("userId must be an integer")
    token = create_token(user_id)
    return jsonify({"token": token})

@app.route("/match/calculate", methods=["POST"])
@auth_required
def match():
    data = request.get_json()
    req = validate_match_request(data)
    try:
        recommendations = get_recommendations(req.studentId)
    except ValueError as e:
        raise NotFound(str(e))
    rec_objs = [Recommendation(**r).dict() for r in recommendations]
    return jsonify(rec_objs)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
