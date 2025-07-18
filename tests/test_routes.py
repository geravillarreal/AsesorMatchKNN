import importlib
import threading
import time
import requests
from werkzeug.serving import make_server

from .conftest import patch_dependencies

class ServerThread(threading.Thread):
    def __init__(self, app, host="127.0.0.1", port=5001):
        super().__init__()
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()
        self.host = host
        self.port = port

    def run(self):
        self.srv.serve_forever()

    def shutdown(self):
        self.srv.shutdown()
        self.ctx.pop()


def start_app():
    with patch_dependencies():
        app_mod = importlib.import_module('app')
    return app_mod


def test_match_endpoint():
    app_mod = start_app()

    # patch get_recommendations
    app_mod.get_recommendations = lambda x: [{"advisorId": 1, "name": "a", "score": 0.9}]

    server = ServerThread(app_mod.app)
    server.start()
    time.sleep(0.2)
    try:
        token = requests.post(f"http://{server.host}:{server.port}/login", json={"userId": 1}).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.post(
            f"http://{server.host}:{server.port}/match/calculate",
            json={"studentId": 1},
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json() == [{"advisorId": 1, "name": "a", "score": 0.9}]
    finally:
        server.shutdown()
        server.join()


def test_match_bad_request():
    app_mod = start_app()
    server = ServerThread(app_mod.app)
    server.start()
    time.sleep(0.2)
    try:
        token = requests.post(f"http://{server.host}:{server.port}/login", json={"userId": 1}).json()["token"]
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.post(
            f"http://{server.host}:{server.port}/match/calculate",
            json={},
            headers=headers,
        )
        assert resp.status_code == 400
    finally:
        server.shutdown()
        server.join()
