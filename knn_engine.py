from __future__ import annotations

import os
import unicodedata
from functools import lru_cache
from typing import Any, Dict, List

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pymysql  # type: ignore
from sentence_transformers import SentenceTransformer, util  # type: ignore
from symspellpy import SymSpell, Verbosity  # type: ignore

# -------------------------------------------------------------
# Models & constants
# -------------------------------------------------------------
_MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
_SYM = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
_SYM.load_dictionary("frequency_dictionary_en_82_765.txt", 0, 1)
_SYM.load_dictionary("frequency_dictionary_es_50k.txt", 0, 1)

_SIM_TH: float = 0.80  # semantic threshold
_WEIGHTS: Dict[str, float] = {"topics": 0.70, "availability": 0.15, "language": 0.15}

_DB_CFG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", "admin123"),
    "db": os.getenv("DB_NAME", "asesorapp"),
    "cursorclass": pymysql.cursors.DictCursor,
}

# -------------------------------------------------------------
# Normalisation helpers
# -------------------------------------------------------------

def _strip_accents(txt: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", txt) if not unicodedata.combining(c)
    )


def _norm(tok: str) -> str:
    tok = _strip_accents(tok.strip().lower())
    sugg = _SYM.lookup(tok, Verbosity.CLOSEST, max_edit_distance=2)
    if sugg:
        tok = sugg[0].term
    return tok


@lru_cache(maxsize=10_000)
def _emb(term: str) -> np.ndarray:  # noqa: D401
    return _MODEL.encode(term, normalize_embeddings=True)


def _canon(lst: List[str]) -> List[str]:
    """Deduplicate list using semantic similarity."""
    out: List[str] = []
    for raw in lst:
        t = _norm(raw)
        if not t:
            continue
        if not any(util.cos_sim(_emb(t), _emb(u)).item() >= 0.99 for u in out):
            out.append(t)
    return out

# -------------------------------------------------------------
# Similarity helpers
# -------------------------------------------------------------

def _coverage(student: List[str], advisor: List[str]) -> float:
    """|A∩B| / |B| with lexical and semantic matching."""
    if not student or not advisor:
        return 0.0

    s_tokens = student
    s_embs = [_emb(t) for t in s_tokens]
    hits = 0
    for adv_tok in advisor:
        # lexical containment (e.g. "quimica" in "quimica organica")
        if any(adv_tok in s or s in adv_tok for s in s_tokens):
            hits += 1
            continue
        # semantic containment
        if util.cos_sim(_emb(adv_tok), s_embs).max().item() >= _SIM_TH:
            hits += 1
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

    areas = _canon(
        _q("SELECT areas FROM profile_areas WHERE profile_id=%s", "areas")
    )
    interests = _canon(
        _q("SELECT interests FROM profile_interests WHERE profile_id=%s", "interests")
    )
    availability = _canon(
        _q("SELECT availability FROM profile_availability WHERE profile_id=%s", "availability")
    )
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

def _list_match(a: List[str], b: List[str]) -> bool:
    return bool(a and b and any(x == y for x in a for y in b))


def _score(st: Dict[str, Any], ad: Dict[str, Any]) -> float:
    cov = _coverage(st["topics"], ad["topics"])
    if cov == 0:
        return 0.0  # skip unrelated advisor

    avail = 1.0 if _list_match(st["availability"], ad["availability"]) else 0.0
    lang = 1.0 if _list_match(st["language"], ad["language"]) else 0.0

    return (
        cov * _WEIGHTS["topics"]
        + avail * _WEIGHTS["availability"]
        + lang * _WEIGHTS["language"]
    )

# -------------------------------------------------------------
# Plot helper
# -------------------------------------------------------------

def _plot(title: str, ranked: List[Dict[str, Any]]):  # noqa: D401
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
    con = pymysql.connect(**_DB_CFG)
    try:
        with con.cursor() as cur:
            student = _load_profile(cur, student_id)
            cur.execute("SELECT id FROM users WHERE role='ADVISOR'")
            advisor_ids = [r["id"] for r in cur.fetchall()]
            advisors = [_load_profile(cur, aid) for aid in advisor_ids]
    finally:
        con.close()

    ranked = [
        {
            "advisorId": adv["user_id"],
            "name": adv["name"],
            "score": round(_score(student, adv), 4),
        }
        for adv in advisors
        if (s := _score(student, adv)) > 0
    ]
    ranked.sort(key=lambda x: x["score"], reverse=True)
    ranked = ranked[: top_k]

    if plot and ranked:
        _plot(student["name"], ranked)

    return ranked
