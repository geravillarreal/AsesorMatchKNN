# knn_engine.py — KNN por tokens (áreas+intereses) desde BD MySQL
from __future__ import annotations

import os
from typing import Dict, List, Tuple, Iterable

import numpy as np


# =========================
# Config DB (env vars)
# =========================
_DB_CFG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", "admin123"),
    "db": os.getenv("DB_NAME", "asesorapp"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "charset": "utf8mb4",
}

# top-k por defecto
_DEFAULT_TOP_K = int(os.getenv("TOP_K_DEFAULT", "5"))

# boosts opcionales por disponibilidad/idioma (binarios, pequeños)
_WEIGHT_AVAIL = float(os.getenv("WEIGHT_AVAIL_BIN", "0.05"))   # 0.0–0.2
_WEIGHT_LANG  = float(os.getenv("WEIGHT_LANG_BIN", "0.05"))    # 0.0–0.2

# Filtro estricto: requiere al menos N tokens exactos en común
_MIN_TOPIC_OVERLAP = int(os.getenv("MIN_TOPIC_OVERLAP", "1"))

# ===== Sinónimos (para fallback) =====
# Ajusta/libra aquí según tu dominio; todos se normalizan (minúsculas, sin acentos).
_SYN_GROUPS: Dict[str, List[str]] = {
    "ia": ["inteligencia artificial", "machine learning", "ml", "aprendizaje automatico"],
    "cloud": ["nube", "computacion en la nube", "aws", "azure", "gcp"],
    "design": ["diseño", "ux", "ui", "ux/ui", "uxui"],
    "programacion": ["desarrollo", "desarrollo de software", "software", "coding"],
    "java": ["java se", "spring", "spring boot"],  # ajusta si "spring" te resulta muy amplio
}
_USE_FALLBACK_SYNONYMS = os.getenv("USE_FALLBACK_SYNONYMS", "true").strip().lower() == "true"


# =========================
# Utilidades de texto
# =========================
def _strip_accents(txt: str) -> str:
    import unicodedata as _ud
    return "".join(c for c in _ud.normalize("NFKD", txt or "") if not _ud.combining(c))

def _norm_token(tok: str) -> str:
    return _strip_accents((tok or "").strip().lower())

def _dedup_preserve(seq: Iterable[str]) -> List[str]:
    seen = set(); out = []
    for t in seq:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

# Mapa inverso: término -> canon del grupo (p.ej. "azure" -> "cloud")
def _build_syn_inverse() -> Dict[str, str]:
    inv = {}
    for canon, vars_ in _SYN_GROUPS.items():
        c = _norm_token(canon)
        inv[c] = c
        for v in vars_:
            inv[_norm_token(v)] = c
    return inv

_SYN_INV = _build_syn_inverse()

def _expand_with_synonyms(tokens: Iterable[str]) -> List[str]:
    """
    Expande una lista de tokens con su grupo de sinónimos (p.ej. "cloud" añade azure/aws/gcp...).
    """
    base = [_norm_token(t) for t in tokens or []]
    out = set(base)
    for t in base:
        canon = _SYN_INV.get(t)
        if canon:
            out.add(canon)
            for v in _SYN_GROUPS.get(canon, []):
                out.add(_norm_token(v))
    return _dedup_preserve(out)

def _map_to_canon(tokens: Iterable[str]) -> List[str]:
    """
    Mapea cada token a su "canon" de sinónimos si existe (p.ej. azure->cloud, ml->ia).
    """
    out = []
    for t in tokens or []:
        t = _norm_token(t)
        out.append(_SYN_INV.get(t, t))
    return _dedup_preserve(out)


# =========================
# Carga de datos desde BD
# (asume esquema: users, profile, profile_areas, profile_interests,
#  profile_availability; language en profile)
# =========================
def _connect():
    import pymysql
    return pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **_DB_CFG)

def _load_student(conn, student_id: int) -> Dict:
    with conn.cursor() as cur:
        cur.execute("SELECT p.id AS pid, u.full_name AS name, p.language AS language "
                    "FROM profile p JOIN users u ON u.id=p.user_id WHERE p.user_id=%s", (student_id,))
        row = cur.fetchone()
        if not row:
            return {"user_id": student_id, "name": f"Student {student_id}",
                    "topics": [], "availability": [], "language": []}

        pid = row["pid"]
        # Áreas
        cur.execute("SELECT areas FROM profile_areas WHERE profile_id=%s", (pid,))
        areas = [_norm_token(r["areas"]) for r in cur.fetchall()]
        # Intereses
        cur.execute("SELECT interests FROM profile_interests WHERE profile_id=%s", (pid,))
        interests = [_norm_token(r["interests"]) for r in cur.fetchall()]
        # Disponibilidad
        cur.execute("SELECT availability FROM profile_availability WHERE profile_id=%s", (pid,))
        availability = [_norm_token(r["availability"]) for r in cur.fetchall()]
        # Idioma
        language = [_norm_token(row["language"])] if row.get("language") else []

        return {
            "user_id": student_id,
            "name": row.get("name") or f"Student {student_id}",
            "topics": _dedup_preserve(areas + interests),
            "availability": _dedup_preserve(availability),
            "language": _dedup_preserve(language),
        }

def _load_advisors(conn) -> List[Dict]:
    advisors: List[Dict] = []
    with conn.cursor() as cur:
        cur.execute("SELECT id, full_name FROM users WHERE role='ADVISOR'")
        users = cur.fetchall()
        if not users:
            return []

        id_map = {}
        for u in users:
            cur.execute("SELECT id, language FROM profile WHERE user_id=%s", (u["id"],))
            pr = cur.fetchone()
            if not pr:
                advisors.append({
                    "user_id": u["id"], "name": u["full_name"] or "",
                    "topics": [], "availability": [], "language": []
                })
                continue
            id_map[u["id"]] = {"pid": pr["id"], "name": u["full_name"], "language": pr.get("language")}

        # Tablas auxiliares completas
        cur.execute("SELECT profile_id, areas FROM profile_areas")
        by_pid_areas: Dict[int, List[str]] = {}
        for r in cur.fetchall():
            by_pid_areas.setdefault(r["profile_id"], []).append(_norm_token(r["areas"]))

        cur.execute("SELECT profile_id, interests FROM profile_interests")
        by_pid_interests: Dict[int, List[str]] = {}
        for r in cur.fetchall():
            by_pid_interests.setdefault(r["profile_id"], []).append(_norm_token(r["interests"]))

        cur.execute("SELECT profile_id, availability FROM profile_availability")
        by_pid_avail: Dict[int, List[str]] = {}
        for r in cur.fetchall():
            by_pid_avail.setdefault(r["profile_id"], []).append(_norm_token(r["availability"]))

        for uid, meta in id_map.items():
            pid = meta["pid"]
            lang = _norm_token(meta["language"]) if meta.get("language") else ""
            advisors.append({
                "user_id": uid,
                "name": meta.get("name") or "",
                "topics": _dedup_preserve((by_pid_areas.get(pid, []) + by_pid_interests.get(pid, []))),
                "availability": _dedup_preserve(by_pid_avail.get(pid, [])),
                "language": _dedup_preserve([lang] if lang else []),
            })

    return advisors


# =========================
# Vectorización binaria
# =========================
def _build_vocab(list_of_token_lists: List[List[str]]) -> List[str]:
    """Vocabulario = unión de tokens (en el orden de aparición)."""
    vocab = []
    seen = set()
    for tokens in list_of_token_lists:
        for t in tokens or []:
            if t and t not in seen:
                seen.add(t); vocab.append(t)
    return vocab

def _to_binary_vec(tokens: Iterable[str], vocab_index: Dict[str, int]) -> np.ndarray:
    v = np.zeros((len(vocab_index),), dtype=np.float32)
    for t in tokens or []:
        idx = vocab_index.get(t)
        if idx is not None:
            v[idx] = 1.0
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


# =========================
# Similitud y KNN
# =========================
def _cosine_topk(query_vec: np.ndarray, mat: np.ndarray, k: int) -> Tuple[List[int], np.ndarray]:
    """Top-k por similitud coseno (producto punto porque ya normalizamos)."""
    if mat.shape[0] == 0:
        return [], np.zeros((0,), dtype=np.float32)
    sims = mat @ query_vec.reshape(-1, 1)  # (N,1)
    order = np.argsort(-sims.flatten())
    idxs = order[:min(k, mat.shape[0])].tolist()
    return idxs, sims.flatten()


# =========================
# API principal (BD)
# =========================
def get_recommendations(student_id: int, top_k: int = None) -> List[Dict]:
    """
    Pipeline:
      A) Estricto por tokens EXACTOS en común (KNN cosine binario). Requiere ≥ _MIN_TOPIC_OVERLAP.
      B) Si faltan candidatos y _USE_FALLBACK_SYNONYMS=True:
         - Fallback por SINÓNIMOS (expansión + cosine).
    Retorna: [{advisorId, name, score, partial?}]
    """
    top_k = top_k or _DEFAULT_TOP_K

    conn = _connect()
    try:
        student = _load_student(conn, student_id)
        advisors = _load_advisors(conn)
    finally:
        conn.close()

    if not advisors or not student.get("topics"):
        return []

    # ---------- A) Estricto (tokens exactos) ----------
    vocab = _build_vocab([ad["topics"] for ad in advisors])
    vocab_index = {t: i for i, t in enumerate(vocab)}
    A = np.vstack([_to_binary_vec(ad["topics"], vocab_index) for ad in advisors])  # (N, V)
    s_vec = _to_binary_vec(student["topics"], vocab_index)
    if np.allclose(s_vec, 0):
        # el alumno no comparte tokens exactos con ningún asesor
        strict_results = []
    else:
        idxs_all, sims = _cosine_topk(s_vec, A, k=max(top_k * 5, top_k))  # pedimos más para filtrar por overlap
        strict_results = []
        for i in idxs_all:
            shared = len(set(student["topics"]) & set(advisors[i]["topics"]))
            if shared < _MIN_TOPIC_OVERLAP:
                continue  # descartar sin temas en común (regla clave)
            base = float(sims[i])
            # boosts binarios pequeños
            if student.get("availability") and advisors[i].get("availability"):
                if any(a in student["availability"] for a in advisors[i]["availability"]):
                    base += _WEIGHT_AVAIL
            if student.get("language") and advisors[i].get("language"):
                if any(l in student["language"] for l in advisors[i]["language"]):
                    base += _WEIGHT_LANG
            strict_results.append({
                "advisorId": advisors[i]["user_id"],
                "name": advisors[i]["name"] or str(advisors[i]["user_id"]),
                "score": round(min(base, 1.0), 4),
                "partial": False,
            })

        strict_results.sort(key=lambda x: x["score"], reverse=True)

    # Si alcanzamos top_k con estricto, listo
    final: List[Dict] = []
    seen = set()
    for r in strict_results:
        if r["advisorId"] in seen:
            continue
        seen.add(r["advisorId"])
        final.append(r)
        if len(final) >= top_k:
            return final

    # ---------- B) Fallback por sinónimos ----------
    if _USE_FALLBACK_SYNONYMS and len(final) < top_k:
        # Expandimos tokens del alumno y asesores
        student_exp = _expand_with_synonyms(student["topics"])
        advisors_exp = [_expand_with_synonyms(ad["topics"]) for ad in advisors]

        # Requerimos al menos 1 coincidencia por grupo de sinónimo (equivalencia por "canon")
        student_canon = set(_map_to_canon(student["topics"]))
        advisors_canon = [set(_map_to_canon(ad["topics"])) for ad in advisors]

        # Vocab expandido y matriz
        vocab2 = _build_vocab(advisors_exp)
        vocab2_index = {t: i for i, t in enumerate(vocab2)}
        A2 = np.vstack([_to_binary_vec(tokens, vocab2_index) for tokens in advisors_exp])
        s2_vec = _to_binary_vec(student_exp, vocab2_index)

        if not np.allclose(s2_vec, 0):
            idxs2, sims2 = _cosine_topk(s2_vec, A2, k=max(top_k * 8, top_k))
            for i in idxs2:
                if advisors[i]["user_id"] in seen:
                    continue  # ya lo agregamos en estricto
                # filtrar: que comparta al menos UN canon de sinónimo
                if len(student_canon & advisors_canon[i]) == 0:
                    continue

                base = float(sims2[i])
                if student.get("availability") and advisors[i].get("availability"):
                    if any(a in student["availability"] for a in advisors[i]["availability"]):
                        base += _WEIGHT_AVAIL
                if student.get("language") and advisors[i].get("language"):
                    if any(l in student["language"] for l in advisors[i]["language"]):
                        base += _WEIGHT_LANG

                final.append({
                    "advisorId": advisors[i]["user_id"],
                    "name": advisors[i]["name"] or str(advisors[i]["user_id"]),
                    "score": round(min(base, 1.0), 4),
                    "partial": True,  # << marca como “coincidencia parcial (por sinónimos)”
                })
                seen.add(advisors[i]["user_id"])
                if len(final) >= top_k:
                    break

    # Orden final por score descendente
    final.sort(key=lambda x: x["score"], reverse=True)
    return final[:top_k]
