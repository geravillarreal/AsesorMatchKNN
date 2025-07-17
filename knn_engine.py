import pymysql
import numpy as np
import unicodedata
from functools import lru_cache
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from symspellpy import SymSpell, Verbosity
import matplotlib.pyplot as plt

_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
_sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
_sym.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)
_sym.load_dictionary("frequency_dictionary_es_82_765.txt", 0, 1)

_WEIGHTS = {"topics": 0.7, "availability": 0.15, "language": 0.15}

DB_CFG = {
    "host": "localhost",
    "user": "root",
    "password": "admin123",
    "db": "asesorapp",
    "cursorclass": pymysql.cursors.DictCursor,
}

# ------------------------------------------------------------------
# Text normalisation helpers
# ------------------------------------------------------------------

def _strip_accents(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", txt) if not unicodedata.combining(c))


def _norm_token(tok: str) -> str:
    t = _strip_accents(tok.strip().lower())
    sug = _sym.lookup(t, Verbosity.CLOSEST, max_edit_distance=2)
    if sug:
        t = sug[0].term
    return t

# ------------------------------------------------------------------
# Embedding cache
# ------------------------------------------------------------------

@lru_cache(maxsize=10000)
def _embed(term: str):
    return _model.encode(term, normalize_embeddings=True)

# ------------------------------------------------------------------
# Canonicalise list (dedup internal synonyms)
# ------------------------------------------------------------------

def _canon(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for raw in tokens:
        t = _norm_token(raw)
        if not t:
            continue
        if not any(util.cos_sim(_embed(t), _embed(u)).item() >= 0.85 for u in out):
            out.append(t)
    return out

# ------------------------------------------------------------------
# Semantic overlap (each token in A finds closest sem‑sim token in B)
# ------------------------------------------------------------------

def _semantic_overlap(a: List[str], b: List[str], thr: float = 0.85) -> float:
    if not a:
        return 0.0
    emb_b = np.stack([_embed(t) for t in b]) if b else np.empty((0, _embed(" ").shape[0]))
    hits = 0
    for tok in a:
        e = _embed(tok)
        if emb_b.size and util.cos_sim(e, emb_b).max().item() >= thr:
            hits += 1
    return hits / len(a)

# ------------------------------------------------------------------
# Profile helpers
# ------------------------------------------------------------------

def _topics(p: Dict[str, Any]) -> List[str]:
    return _canon(p.get("areas", []) + p.get("interests", []))


def _score(s: Dict[str, Any], t: Dict[str, Any]) -> float:
    sims = {
        "topics": _semantic_overlap(_topics(s), _topics(t)),
        "availability": _semantic_overlap(s.get("availability", []), t.get("availability", []), 1.0),
        "language": _semantic_overlap([_norm_token(s.get("language", ""))], [_norm_token(t.get("language", ""))], 1.0),
    }
    return sum(_WEIGHTS[k] * sims[k] for k in _WEIGHTS)

# ------------------------------------------------------------------
# DB helpers
# ------------------------------------------------------------------

def _conn():
    return pymysql.connect(**DB_CFG)


def _load_profile(cur, uid: int) -> Dict[str, Any]:
    cur.execute("SELECT * FROM profile WHERE user_id=%s", uid)
    prof = cur.fetchone() or {}

    def q(sql: str, col: str) -> List[str]:
        cur.execute(sql, prof.get("id"))
        return [r[col] for r in cur.fetchall()]

    prof["areas"] = q("SELECT areas FROM profile_areas WHERE profile_id=%s", "areas")
    prof["interests"] = q("SELECT interests FROM profile_interests WHERE profile_id=%s", "interests")
    prof["availability"] = q("SELECT availability FROM profile_availability WHERE profile_id=%s", "availability")
    cur.execute("SELECT full_name, faculty FROM users WHERE id=%s", uid)
    u = cur.fetchone() or {}
    prof.update({"name": u.get("full_name", ""), "faculty": u.get("faculty", ""), "user_id": uid})
    
    return prof

# ------------------------------------------------------------------
# Optional plot
# ------------------------------------------------------------------

def _plot(student_name: str, ranked: List[Dict[str, Any]]):
    names = [student_name] + [r["name"] for r in ranked]
    scores = [1.0] + [r["score"] for r in ranked]
    fig, ax = plt.subplots(figsize=(6, 3 + 0.5 * len(names)))
    ax.barh(range(len(names)), scores)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Similarity")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    plt.show()

# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def get_recommendations(student_id: int, top_k: int = 5, plot: bool = False) -> List[Dict[str, Any]]:
    con = _conn()
    try:
        with con.cursor() as cur:
            student = _load_profile(cur, student_id)
            cur.execute("SELECT id FROM users WHERE role='ADVISOR'")
            advisor_ids = [r["id"] for r in cur.fetchall()]
            advisors = [_load_profile(cur, aid) for aid in advisor_ids]
        ranked = sorted(
            (
                {
                    "advisorId": a["user_id"],
                    "name": a["name"],
                    "score": round(_score(student, a), 4),
                }
                for a in advisors
            ),
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]
        if plot:
            _plot(student["name"], ranked)
        return ranked
    finally:
        con.close()
