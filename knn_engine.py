import pymysql
import numpy as np
import unicodedata
from functools import lru_cache
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
from symspellpy import SymSpell, Verbosity
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Models & dictionaries
# -------------------------------------------------------------
_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
_sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
_sym.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)
_sym.load_dictionary("frequency_dictionary_es_50k.txt", 0, 1)

_WEIGHTS = {"topics": 0.7, "availability": 0.15, "language": 0.15}

DB_CFG = {
    "host": "localhost",
    "user": "root",
    "password": "admin123",
    "db": "asesorapp",
    "cursorclass": pymysql.cursors.DictCursor,
}

# -------------------------------------------------------------
# Text normalisation & embedding
# -------------------------------------------------------------

def _strip_accents(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", txt) if not unicodedata.combining(c))


def _norm(tok: str) -> str:
    tok = _strip_accents(tok.strip().lower())
    sugg = _sym.lookup(tok, Verbosity.CLOSEST, max_edit_distance=2)
    if sugg:
        tok = sugg[0].term
    return tok


@lru_cache(maxsize=10000)
def _emb(term: str):
    return _model.encode(term, normalize_embeddings=True)


# deduplicate tokens inside a list using semantic similarity

def _canon(lst: List[str]) -> List[str]:
    out: List[str] = []
    for raw in lst:
        t = _norm(raw)
        if not t:
            continue
        if not any(util.cos_sim(_emb(t), _emb(u)).item() >= 0.85 for u in out):
            out.append(t)
    return out

# -------------------------------------------------------------
# Similarity helpers
# -------------------------------------------------------------

def _coverage(student: List[str], advisor: List[str]) -> float:
    if not student or not advisor:
        return 0.0
    emb_s = [_emb(t) for t in student]            # una sola vez
    hits = sum(
        1
        for tok in advisor
        if util.cos_sim(_emb(tok), emb_s).max().item() >= 0.85
    )
    return hits / len(advisor)

# -------------------------------------------------------------
# Profile construction
# -------------------------------------------------------------

def _load_profile(cur, uid: int) -> Dict[str, Any]:
    cur.execute("SELECT * FROM profile WHERE user_id=%s", uid)
    base = cur.fetchone() or {}

    def _q(sql: str, col: str) -> List[str]:
        cur.execute(sql, base.get("id"))
        return [r[col] for r in cur.fetchall()]

    areas = _canon(_q("SELECT areas FROM profile_areas WHERE profile_id=%s", "areas"))
    interests = _canon(_q("SELECT interests FROM profile_interests WHERE profile_id=%s", "interests"))
    availability = _canon(_q("SELECT availability FROM profile_availability WHERE profile_id=%s", "availability"))
    language = [_norm(base.get("language", ""))] if base.get("language") else []

    cur.execute("SELECT full_name FROM users WHERE id=%s", uid)
    name_row = cur.fetchone() or {}

    return {
        "user_id": uid,
        "name": name_row.get("full_name", ""),
        "topics": sorted(set(areas + interests)),
        "availability": availability,
        "language": language,
    }


# -------------------------------------------------------------
# Scoring
# -------------------------------------------------------------

def _score(st: Dict[str, Any], ad: Dict[str, Any]) -> float:
    cov = _coverage(st["topics"], ad["topics"])
    if cov == 0:
        return 0.0  # ignore completely unrelated advisor
    avail = 1.0 if st["availability"] == ad["availability"] else 0.0
    lang  = 1.0 if st["language"] == ad["language"] else 0.0
    return cov * _WEIGHTS["topics"] + avail * _WEIGHTS["availability"] + lang * _WEIGHTS["language"]



# -------------------------------------------------------------
# Plot helper
# -------------------------------------------------------------

def _plot(title: str, ranked: List[Dict[str, Any]]):
    labels = [title] + [r["name"] or str(r["advisorId"]) for r in ranked]
    vals = [1.0] + [r["score"] for r in ranked]
    plt.figure(figsize=(6, 0.6 * len(labels) + 2))
    plt.barh(labels[::-1], vals[::-1])
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------
# Public API
# -------------------------------------------------------------

def get_recommendations(student_id: int, top_k: int = 5, plot: bool = False):
    con = pymysql.connect(**DB_CFG)
    try:
        with con.cursor() as cur:
            student = _load_profile(cur, student_id)
            cur.execute("SELECT id FROM users WHERE role='ADVISOR'")
            advisor_ids = [r["id"] for r in cur.fetchall()]
            advisors = [_load_profile(cur, aid) for aid in advisor_ids]
    finally:
        con.close()

    ranked = sorted(
        (
            {
                "advisorId": adv["user_id"],
                "name": adv["name"],
                "score": round(_score(student, adv), 4),
            }
            for adv in advisors
        ),
        key=lambda x: x["score"],
        reverse=True,
    )[:top_k]

    if plot:
        _plot(student["name"], ranked)

    return ranked
