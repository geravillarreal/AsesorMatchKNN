import flask
from flask import jsonify
from werkzeug.exceptions import HTTPException, NotFound, BadRequest

from openapi_spec import OPENAPI_SPEC

from knn_engine import get_recommendations
from middleware import log_request
from validation import validate_match_request, Recommendation
from auth import auth_required, create_token

app = flask.Flask(__name__)

app.before_request(log_request)


@app.route("/", methods=["GET"])
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset='utf-8'>
        <title>AsesorMatch UI</title>
    </head>
    <body>
        <h1>AsesorMatch</h1>
        <div>
            <h2>Login</h2>
            <input id='userId' placeholder='User ID'>
            <button onclick='login()'>Login</button>
            <div id='loginStatus'></div>
        </div>
        <div>
            <h2>Calcular recomendaciones</h2>
            <input id='studentId' placeholder='Student ID'>
            <button onclick='calculate()'>Calcular</button>
            <pre id='results'></pre>
        </div>
        <script>
        let token = '';
        function login() {
            const uid = document.getElementById('userId').value;
            fetch('/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({userId: parseInt(uid)})
            }).then(r => r.json())
              .then(d => { token = d.token; document.getElementById('loginStatus').innerText = 'Token obtenido'; })
              .catch(() => alert('Error en login'));
        }
        function calculate() {
            const sid = document.getElementById('studentId').value;
            fetch('/match/calculate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + token
                },
                body: JSON.stringify({studentId: parseInt(sid)})
            }).then(r => r.json())
              .then(d => { document.getElementById('results').innerText = JSON.stringify(d, null, 2); })
              .catch(() => alert('Error en cálculo'));
        }
        </script>
    </body>
    </html>
    """


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
    data = flask.request.get_json() or {}
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
    data = flask.request.get_json()
    req = validate_match_request(data)
    try:
        recommendations = get_recommendations(req.studentId)
    except ValueError as e:
        raise NotFound(str(e))
    rec_objs = [Recommendation(**r).dict() for r in recommendations]
    return jsonify(rec_objs)


@app.route("/openapi.json", methods=["GET"])
def openapi_json():
    """Return OpenAPI specification."""
    return jsonify(OPENAPI_SPEC)


@app.route("/docs", methods=["GET"])
def docs():
    """Serve Swagger UI page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Docs</title>
        <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui.css\" />
    </head>
    <body>
        <div id=\"swagger-ui\"></div>
        <script src=\"https://cdnjs.cloudflare.com/ajax/libs/swagger-ui/4.15.5/swagger-ui-bundle.js\"></script>
        <script>
        SwaggerUIBundle({url:'/openapi.json', dom_id:'#swagger-ui'});
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=True, port=8000)
