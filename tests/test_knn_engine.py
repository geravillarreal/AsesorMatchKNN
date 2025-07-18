import importlib
import numpy as np
import pytest

from .conftest import patch_dependencies


def load_module():
    with patch_dependencies():
        return importlib.import_module('knn_engine')


def test_list_match():
    ke = load_module()
    assert ke._list_match(['a'], ['b', 'a'])
    assert not ke._list_match(['x'], ['y'])


def test_score(monkeypatch):
    ke = load_module()

    def fake_cov(st, adv):
        return 0.5

    monkeypatch.setattr(ke, '_coverage', fake_cov)
    st = {'topics': ['a'], 'availability': ['mon'], 'language': ['en']}
    adv = {'topics': ['b'], 'availability': ['mon'], 'language': ['es']}
    expected = 0.5 * ke._WEIGHTS['topics'] + 1.0 * ke._WEIGHTS['availability'] + 0.0 * ke._WEIGHTS['language']
    assert ke._score(st, adv) == expected


def test_coverage(monkeypatch):
    ke = load_module()

    def fake_emb(term):
        mapping = {'a': np.array([1, 0]), 'b': np.array([0, 1])}
        return mapping[term]

    monkeypatch.setattr(ke, '_emb', fake_emb)

    student = ['a']
    advisor = ['a', 'b']
    cov = ke._coverage(student, advisor)
    assert cov == 0.5
