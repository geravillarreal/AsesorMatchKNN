import sys
import types

class DummyNumpy:
    @staticmethod
    def array(lst):
        return list(lst)

    @staticmethod
    def zeros(n):
        return [0.0] * n

    @staticmethod
    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    @staticmethod
    def atleast_2d(arr):
        return [arr]

numpy_mod = types.ModuleType('numpy')
numpy_mod.array = DummyNumpy.array
numpy_mod.zeros = DummyNumpy.zeros
numpy_mod.dot = DummyNumpy.dot
numpy_mod.atleast_2d = DummyNumpy.atleast_2d
sys.modules['numpy'] = numpy_mod
np = numpy_mod

requests_mod = types.ModuleType('requests')
class Response:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data
    def json(self):
        return self._data

def post(url, json=None, headers=None):
    import urllib.request, json as _json
    data = None
    if json is not None:
        data = _json.dumps(json).encode()
    req = urllib.request.Request(url, data=data, headers=headers or {}, method='POST')
    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.getcode()
            body = resp.read()
            try:
                parsed = _json.loads(body.decode())
            except Exception:
                parsed = None
            return Response(status, parsed)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            parsed = _json.loads(body.decode())
        except Exception:
            parsed = None
        return Response(e.code, parsed)

requests_mod.post = post
sys.modules['requests'] = requests_mod

werk_mod = types.ModuleType('werkzeug')
serving_mod = types.ModuleType('werkzeug.serving')
def make_server(host, port, app):
    from wsgiref.simple_server import make_server as _make_server
    return _make_server(host, port, app)
serving_mod.make_server = make_server
exceptions_mod = types.ModuleType('werkzeug.exceptions')
class HTTPException(Exception):
    def __init__(self, description=None, code=None):
        super().__init__(description)
        self.description = description
        self.code = code
class BadRequest(HTTPException):
    pass
class NotFound(HTTPException):
    pass
class Unauthorized(HTTPException):
    pass
exceptions_mod.HTTPException = HTTPException
exceptions_mod.BadRequest = BadRequest
exceptions_mod.NotFound = NotFound
exceptions_mod.Unauthorized = Unauthorized
werk_mod.serving = serving_mod
werk_mod.exceptions = exceptions_mod
sys.modules['werkzeug'] = werk_mod
sys.modules['werkzeug.serving'] = serving_mod
sys.modules['werkzeug.exceptions'] = exceptions_mod
from contextlib import contextmanager

@contextmanager
def patch_dependencies():
    """Patch heavy dependencies used by knn_engine and app."""
    st_mod = types.ModuleType('sentence_transformers')
    class DummyModel:
        def encode(self, term, normalize_embeddings=True):
            mapping = {
                'a': np.array([1.0, 0.0]),
                'b': np.array([0.0, 1.0]),
                'c': np.array([1.0, 1.0]),
            }
            return mapping.get(term, np.zeros(2))
    def cos_sim(a, b):
        b = np.atleast_2d(b)
        sims = np.dot(b, a)
        return np.array([sims])
    st_mod.SentenceTransformer = lambda *a, **k: DummyModel()
    st_mod.util = types.SimpleNamespace(cos_sim=cos_sim)
    sys.modules['sentence_transformers'] = st_mod

    sym_mod = types.ModuleType('symspellpy')
    class DummySymSpell:
        def __init__(self, *a, **k):
            pass
        def load_dictionary(self, *a, **k):
            pass
        def lookup(self, *a, **k):
            return []
    sym_mod.SymSpell = DummySymSpell
    sym_mod.Verbosity = types.SimpleNamespace(CLOSEST=0)
    sys.modules['symspellpy'] = sym_mod

    py_mod = types.ModuleType('pymysql')
    py_mod.cursors = types.SimpleNamespace(DictCursor=object)
    py_mod.connect = lambda **k: None
    sys.modules['pymysql'] = py_mod

    import urllib.request
    def noop(*a, **k):
        return ('', None)
    urllib.request.urlretrieve = noop

    # Stub pydantic BaseModel for tests
    pyd_mod = types.ModuleType('pydantic')
    class BaseModel:
        def __init__(self, **data):
            for field, typ in self.__annotations__.items():
                if field not in data:
                    raise ValueError(f"{field} field required")
                val = data[field]
                if typ is int and not isinstance(val, int):
                    raise ValueError(f"{field} must be int")
                if typ is float and not isinstance(val, (int, float)):
                    raise ValueError(f"{field} must be float")
                if typ is str and not isinstance(val, str):
                    raise ValueError(f"{field} must be str")
                setattr(self, field, val)
        def dict(self):
            return {f: getattr(self, f) for f in self.__annotations__}
    pyd_mod.BaseModel = BaseModel
    sys.modules['pydantic'] = pyd_mod

    # Stub jwt module for tests
    jwt_mod = types.ModuleType('jwt')
    def encode(payload, secret, algorithm=None):
        return 'token'
    def decode(token, secret, algorithms=None):
        return {'user_id': 1}
    jwt_mod.encode = encode
    jwt_mod.decode = decode
    jwt_mod.ExpiredSignatureError = Exception
    jwt_mod.InvalidTokenError = Exception
    sys.modules['jwt'] = jwt_mod


    try:
        yield
    finally:
        pass
