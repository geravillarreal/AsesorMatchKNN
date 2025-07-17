import pymysql
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import ParameterGrid

# Multilingual sentence embeddings for automatic synonym / translation handling
from sentence_transformers import SentenceTransformer
from functools import lru_cache

"""
KNN‑BASED ADVISOR RECOMMENDER – SEMANTIC VERSION (safe for empty lists)
======================================================================
* Detecta sinónimos / traducciones con embeddings multilingües.
* Corrige error: evita llamar a cosine_similarity con matrices vacías
  (tokens únicos ⇒ shape (0, 384)).
* API sin cambios: `get_recommendations(student_id, top_k=5)`.
"""

# ───────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────
DB_CONFIG = dict(
    host="localhost",
    user="root",
    password="admin123",
    db="asesorapp",
    cursorclass=pymysql.cursors.DictCursor,
)

ATTRS: Tuple[str, ...] = (
    "areas",
    "interests",
    "language",
    "availability",
    "books",
    "faculty",
    "modality",
    "level",
)

DEFAULT_WEIGHTS: Dict[str, float] = {
    "areas":        0.30,
    "interests":    0.25,
    "language":     0.15,
    "availability": 0.10,
    "books":        0.10,
    "faculty":      0.05,
    "modality":     0.03,
    "level":        0.02,
}

TOP_K_DEFAULT: int = 5
SIM_THRESHOLD: float = 0.80  # ≥ threshold ⇒ treat words as synonyms

# ───────────────────────────────────────────────
# EMBEDDING MODEL (lazy)
# ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_emb_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


def _embed_many(tokens: List[str]) -> np.ndarray:
    if not tokens:
        return np.zeros((0, 384))
    return _get_emb_model().encode(tokens, show_progress_bar=False, normalize_embeddings=True)

# ───────────────────────────────────────────────
# DB UTILITIES
# ───────────────────────────────────────────────

def _get_connection():
    return pymysql.connect(**DB_CONFIG)

# ───────────────────────────────────────────────
# PROFILE LOADING
# ───────────────────────────────────────────────

def _load_profile(cur, user_id: int) -> Dict[str, Any] | None:
    cur.execute("SELECT * FROM profile WHERE user_id=%s", user_id)
    p = cur.fetchone()
    if p is None:
        return None

    def col(sql: str, col_name: str) -> List[str]:
        cur.execute(sql, p["id"])
        return [r[col_name] for r in cur.fetchall()]

    p["areas"]        = col("SELECT areas FROM profile_areas WHERE profile_id=%s", "areas")
    p["interests"]    = col("SELECT interests FROM profile_interests WHERE profile_id=%s", "interests")
    p["availability"] = col("SELECT availability FROM profile_availability WHERE profile_id=%s", "availability")
    p["books"] = [t.lower() for t in col("SELECT title FROM book WHERE profile_id=%s", "title")]

    cur.execute("SELECT full_name, faculty FROM users WHERE id=%s", user_id)
    u = cur.fetchone() or {}
    p["name"]    = u.get("full_name", "NoName")
    p["faculty"] = u.get("faculty", "Undefined")
    p["user_id"] = user_id

    for k, v in {"language": "Desconocido", "level": "Desconocido", "modality": "Desconocido"}.items():
        p.setdefault(k, v)
    for k in ("areas", "interests", "availability", "books"):
        p[k] = p.get(k) or []
    return p

# ───────────────────────────────────────────────
# SYNONYM CLUSTERING & CANONICALISATION
# ───────────────────────────────────────────────

def _cluster_tokens(tokens: List[str], threshold: float = SIM_THRESHOLD) -> Tuple[Dict[str, str], List[str]]:
    """Groups tokens into synonym clusters and returns mapping + canonical list."""
    if not tokens:
        return {}, []
    vectors = _embed_many(tokens)
    canonical: List[str] = []
    mapping: Dict[str, str] = {}
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if tok in mapping:
            continue  # already mapped to a canonical token
        canonical.append(tok)
        mapping[tok] = tok
        if i == n - 1:
            continue  # last token ⇒ no remaining tokens to compare
        rest = vectors[i + 1 :]
        if rest.shape[0] == 0:
            continue  # safety: no vectors left
        sims = cosine_similarity(vectors[i].reshape(1, -1), rest)[0]
        for j, sim in enumerate(sims, start=i + 1):
            if sim >= threshold:
                mapping[tokens[j]] = tok
    return mapping, canonical


def _canonicalise_profiles(profiles: List[Dict[str, Any]], syn_maps: Dict[str, Dict[str, str]]):
    for p in profiles:
        for key in ATTRS:
            if key not in p:
                continue
            if isinstance(p[key], list):
                p[key] = list({syn_maps[key].get(t.lower(), t.lower()) for t in p[key]})
            else:
                p[key] = syn_maps[key].get(str(p[key]).lower(), str(p[key]).lower())

# ───────────────────────────────────────────────
# FEATURE SPACES (canonical tokens)
# ───────────────────────────────────────────────

def _build_spaces(profiles: List[Dict[str, Any]]):
    raw = {
        "areas":        sorted({t.lower() for p in profiles for t in p["areas"]}),
        "interests":    sorted({t.lower() for p in profiles for t in p["interests"]}),
        "availability": sorted({t.lower() for p in profiles for t in p["availability"]}),
        "language":     sorted({p["language"].lower() for p in profiles}),
        "modality":     sorted({p["modality"].lower() for p in profiles}),
        "level":        sorted({p["level"].lower() for p in profiles}),
        "faculty":      sorted({p["faculty"].lower() for p in profiles}),
        "books":        sorted({w for p in profiles for t in p["books"] for w in t.split() if len(w) > 3}),
    }
    syn_maps, spaces = {}, {}
    for key, tokens in raw.items():
        mp, canon = _cluster_tokens(tokens)
        syn_maps[key] = mp
        spaces[key] = canon
    _canonicalise_profiles(profiles, syn_maps)
    return spaces

# ───────────────────────────────────────────────
# VECTORIZATION (binary)
# ───────────────────────────────────────────────

def _vectorize(p: Dict[str, Any], space: List[str], key: str) -> np.ndarray:
    if key in ("language", "modality", "level", "faculty"):
        return np.array([1 if p[key] == t else 0 for t in space])
    if key == "books":
        text = " ".join(p[key])
        return np.array([1 if t in text else 0 for t in space])
    return np.array([1 if t in p[key] else 0 for t in space])

# ───────────────────────────────────────────────
# SIMILARITY
# ───────────────────────────────────────────────

def _similarity(p1: Dict[str, Any], p2: Dict[str, Any], spaces: Dict[str, List[str]], wts: Dict[str, float]) -> float:
    score = 0.0
    for k, w in wts.items():
        v1, v2 = _vectorize(p1, spaces[k], k), _vectorize(p2, spaces[k], k)
        if not np.any(v1) or not np.any(v2):
            continue
        score += w * cosine_similarity([v1], [v2])[0][0]
    return score

# ───────────────────────────────────────────────
# MAIN ENTRY POINT
# ───────────────────────────────────────────────

def get_recommendations(student_id: int, top_k: int = TOP_K_DEFAULT, weights: Dict[str, float] | None = None):
    weights = weights or DEFAULT_WEIGHTS
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            student = _load_profile(cur, student_id)
            if student is None:
                return []
            cur.execute("SELECT id FROM users WHERE role='ADVISOR'")
            advisors = [_load_profile(cur, r["id"]) for r in cur.fetchall()]
            advisors = [a for a in advisors if a]
        if not advisors:
            return []
        spaces = _build_spaces(advisors + [student])
        sims = [_similarity(student, adv, spaces, weights) for adv in advisors]
        ranked = sorted(
            ({"advisorId": adv["user_id"], "name": adv["name"], "faculty": adv["faculty"], "score": round(s, 4)}
             for adv, s in zip(advisors, sims)),
            key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]
    finally:
        conn.close()