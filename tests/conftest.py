import sys
import types
import numpy as np
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

    try:
        yield
    finally:
        pass
