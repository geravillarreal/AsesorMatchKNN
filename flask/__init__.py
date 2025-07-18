import json
from types import SimpleNamespace

request = SimpleNamespace()

class Request:
    def __init__(self, environ):
        self.method = environ.get('REQUEST_METHOD', 'GET')
        self.path = environ.get('PATH_INFO', '/')
        self.headers = {
            k[5:].replace('_', '-').title(): v
            for k, v in environ.items()
            if k.startswith('HTTP_')
        }
        self.environ = environ

    def get_json(self):
        length = int(self.environ.get('CONTENT_LENGTH') or 0)
        if length:
            data = self.environ['wsgi.input'].read(length)
            if data:
                return json.loads(data.decode())
        return None


class Flask:
    def __init__(self, name: str):
        self.name = name
        self.routes = {}
        self.before_fn = None
        self.error_handlers = {}
        self._ctx_stack = []

    def before_request(self, func):
        self.before_fn = func

    def route(self, path, methods=None):
        if methods is None:
            methods = ['GET']

        def decorator(func):
            for m in methods:
                self.routes[(path, m)] = func
            return func

        return decorator

    def errorhandler(self, exc):
        def decorator(func):
            self.error_handlers[exc] = func
            return func

        return decorator

    def run(self, debug=False, port=5000):
        from wsgiref.simple_server import make_server

        srv = make_server('0.0.0.0', port, self)
        srv.serve_forever()

    def app_context(self):
        class Ctx:
            def __init__(self, app):
                self.app = app
            def push(self):
                self.app._ctx_stack.append(self)
            def pop(self):
                if self.app._ctx_stack:
                    self.app._ctx_stack.pop()
        return Ctx(self)

    def __call__(self, environ, start_response):
        global request
        request = Request(environ)
        if self.before_fn:
            self.before_fn()
        handler = self.routes.get((request.path, request.method))
        if handler is None:
            start_response('404 NOT FOUND', [('Content-Type', 'application/json')])
            return [b'{"error":"Not Found"}']
        try:
            resp = handler()
        except Exception as e:
            for exc, h in self.error_handlers.items():
                if isinstance(e, exc):
                    resp = h(e)
                    break
            else:
                resp = ({'error': 'Internal server error', 'detail': str(e)}, 500)
        if isinstance(resp, tuple):
            body, status = resp
        else:
            body, status = resp, 200
        if isinstance(body, (dict, list)):
            body_bytes = json.dumps(body).encode()
            headers = [('Content-Type', 'application/json')]
        elif isinstance(body, bytes):
            body_bytes = body
            headers = [('Content-Type', 'application/octet-stream')]
        elif isinstance(body, str):
            body_bytes = body.encode()
            if body.lstrip().startswith('<'):
                headers = [('Content-Type', 'text/html')]
            else:
                headers = [('Content-Type', 'text/plain')]
        else:
            body_bytes = str(body).encode()
            headers = [('Content-Type', 'text/plain')]
        start_response(f'{status} OK', headers)
        return [body_bytes]


def jsonify(obj):
    return obj
