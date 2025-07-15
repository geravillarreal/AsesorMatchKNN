from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()

    # Respuesta simulada (mock)
    recommendations = [
        {"advisorId": 1, "name": "Dr. Sergio Alcaraz", "score": 0.93},
        {"advisorId": 2, "name": "Dra. Ana Gómez", "score": 0.87},
        {"advisorId": 3, "name": "Mtro. Luis Ramírez", "score": 0.81}
    ]
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(port=5000)
