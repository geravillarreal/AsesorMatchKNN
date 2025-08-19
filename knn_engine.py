# knn_engine.py — KNN TF-IDF con guardas por dominio (BD MySQL) + fallback semántico opcional
from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

# =========================
# Configuración por env vars
# =========================
_DB_CFG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", "admin123"),
    "db": os.getenv("DB_NAME", "asesorapp"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "charset": "utf8mb4",
}

_DEFAULT_TOP_K = int(os.getenv("TOP_K_DEFAULT", "5"))

# Overlap mínimo para aceptar un candidato en la fase estricta
MIN_TOKEN_OVERLAP = int(os.getenv("MIN_TOKEN_OVERLAP", "1"))      # tokens normalizados exactos
MIN_STEM_OVERLAP  = int(os.getenv("MIN_STEM_OVERLAP", "1"))       # coincidencias por raíces (stems)

# Boosts suaves (0–0.2 es razonable)
WEIGHT_AVAIL = float(os.getenv("WEIGHT_AVAIL_BIN", "0.05"))
WEIGHT_LANG  = float(os.getenv("WEIGHT_LANG_BIN", "0.05"))

# Cache de asesores (segundos)
ADVISOR_CACHE_TTL = int(os.getenv("ADVISOR_CACHE_TTL", "60"))

# Fallback semántico (requiere sentence-transformers)
USE_EMBED_FALLBACK: bool = os.getenv("USE_EMBED_FALLBACK", "false").strip().lower() == "true"
SBERT_MODEL: str = os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
SBERT_LOCAL_PATH: str = os.getenv("SBERT_LOCAL_PATH", "").strip()
SIM_TH_EMB: float = float(os.getenv("SIM_TH_EMB", "0.70"))
MIN_SEMANTIC_HITS: int = int(os.getenv("MIN_SEMANTIC_HITS", "1"))

# Si TRUE, en fallback exigimos por lo menos 1 stem en común además de similitud
FALLBACK_REQUIRE_STEM_OVERLAP: bool = os.getenv("FALLBACK_REQUIRE_STEM_OVERLAP", "true").lower() == "true"

# =========================
# Utils de normalización
# =========================
def _strip_accents(txt: str) -> str:
    import unicodedata as _ud
    return "".join(c for c in _ud.normalize("NFKD", txt or "") if not _ud.combining(c))

def _norm_token(tok: str) -> str:
    t = _strip_accents((tok or "").strip().lower())
    # limpiar separators básicos
    for ch in (",", ";", ".", ":", "/", "\\", "|", "(", ")", "[", "]", "{", "}", "+", "#"):
        t = t.replace(ch, " ")
    t = " ".join(t.split())
    return t

def _simple_stem(word: str) -> str:
    """Stemmer MUY ligero para ES/EN que ayuda a agrupar dominios.
    No dependemos de NLTK. Evita sobre-stemming agresivo."""
    w = _norm_token(word)
    if not w:
        return w
    # plurales y terminaciones comunes (muy conservador)
    for suf in ("mente", "mente", "aciones", "aciones", "cion", "ciones", "siones", "sion",
                "idades", "idad", "mente", "es", "s", "ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    return w

def _dedup_preserve(seq: Iterable[str]) -> List[str]:
    seen = set(); out = []
    for t in seq:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out

# =========================
# Conexión BD y carga de datos
# =========================
def _connect():
    import pymysql
    return pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **_DB_CFG)

def _load_student(conn, student_id: int) -> Dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT p.id AS pid, u.full_name AS name, p.language AS language
            FROM profile p
            JOIN users u ON u.id = p.user_id
            WHERE p.user_id = %s
            """,
            (student_id,),
        )
        row = cur.fetchone()
        if not row:
            # estudiante sin perfil todavía
            return {
                "user_id": student_id,
                "name": f"Student {student_id}",
                "tokens": [],
                "stems": [],
                "availability": [],
                "language": [],
            }

        pid = row["pid"]
        # Áreas / intereses / disponibilidad
        cur.execute("SELECT areas FROM profile_areas WHERE profile_id=%s", (pid,))
        areas = [_norm_token(r["areas"]) for r in cur.fetchall()]

        cur.execute("SELECT interests FROM profile_interests WHERE profile_id=%s", (pid,))
        interests = [_norm_token(r["interests"]) for r in cur.fetchall()]

        cur.execute("SELECT availability FROM profile_availability WHERE profile_id=%s", (pid,))
        availability = [_norm_token(r["availability"]) for r in cur.fetchall()]

        language = [_norm_token(row["language"])] if row.get("language") else []

        # tokens (palabras individuales) a partir de frases de áreas/intereses
        phrases = [p for p in areas + interests if p]
        tokens: List[str] = []
        for ph in phrases:
            tokens.extend([t for t in ph.split(" ") if t])

        tokens = _dedup_preserve(tokens)
        stems = _dedup_preserve(_simple_stem(t) for t in tokens)

        return {
            "user_id": student_id,
            "name": row.get("name") or f"Student {student_id}",
            "tokens": tokens,
            "stems": stems,
            "availability": _dedup_preserve(availability),
            "language": _dedup_preserve(language),
        }

# --- Cache simple de asesores en memoria (TTL) ---
_advisors_cache: Optional[List[Dict]] = None
_advisors_cache_until: float = 0.0

def refresh_advisors_cache() -> None:
    """Permite invalidar cache externamente si haces cambios masivos en la BD."""
    global _advisors_cache_until
    _advisors_cache_until = 0.0

def _load_advisors(conn) -> List[Dict]:
    global _advisors_cache, _advisors_cache_until
    now = time.time()
    if _advisors_cache is not None and now < _advisors_cache_until:
        return _advisors_cache

    advisors: List[Dict] = []
    with conn.cursor() as cur:
        cur.execute("SELECT id, full_name FROM users WHERE role='ADVISOR'")
        users = cur.fetchall()
        if not users:
            _advisors_cache = []
            _advisors_cache_until = now + ADVISOR_CACHE_TTL
            return _advisors_cache

        # cargar profile id + idioma
        id_map = {}
        for u in users:
            cur.execute("SELECT id, language FROM profile WHERE user_id=%s", (u["id"],))
            pr = cur.fetchone()
            if not pr:
                advisors.append({
                    "user_id": u["id"],
                    "name": u.get("full_name") or "",
                    "tokens": [],
                    "stems": [],
                    "availability": [],
                    "language": [],
                })
                continue
            id_map[u["id"]] = {
                "pid": pr["id"],
                "name": u["full_name"],
                "language": pr.get("language"),
            }

        # precargar tablas relacionales
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
            language = _norm_token(meta["language"]) if meta.get("language") else ""
            phrases = (by_pid_areas.get(pid, []) + by_pid_interests.get(pid, []))
            tokens: List[str] = []
            for ph in phrases:
                if not ph:
                    continue
                tokens.extend([t for t in ph.split(" ") if t])
            tokens = _dedup_preserve(tokens)
            stems = _dedup_preserve(_simple_stem(t) for t in tokens)

            advisors.append({
                "user_id": uid,
                "name": meta.get("name") or "",
                "tokens": tokens,
                "stems": stems,
                "availability": _dedup_preserve(by_pid_avail.get(pid, [])),
                "language": _dedup_preserve([language] if language else []),
            })

    _advisors_cache = advisors
    _advisors_cache_until = now + ADVISOR_CACHE_TTL
    return _advisors_cache

# =========================
# TF-IDF y similitudes
# =========================
def _build_vocab(tokens_list: List[List[str]]) -> List[str]:
    vocab = []
    seen = set()
    for tokens in tokens_list:
        for t in tokens or []:
            if t and t not in seen:
                seen.add(t); vocab.append(t)
    return vocab

def _tfidf_matrix(list_of_token_lists: List[List[str]]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Crea TF-IDF simple (sin dependencias externas). Normaliza por L2."""
    vocab = _build_vocab(list_of_token_lists)
    V = len(vocab)
    if V == 0:
        return np.zeros((len(list_of_token_lists), 0), dtype=np.float32), {}
    idx = {t: i for i, t in enumerate(vocab)}

    # Document Frequency
    df = np.zeros((V,), dtype=np.float32)
    for tokens in list_of_token_lists:
        if not tokens:
            continue
        unique = set(tokens)
        for t in unique:
            df[idx[t]] += 1.0

    N = float(len(list_of_token_lists))
    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0  # idf suavizado

    # TF * IDF
    M = np.zeros((len(list_of_token_lists), V), dtype=np.float32)
    for i, tokens in enumerate(list_of_token_lists):
        if not tokens:
            continue
        # tf
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        # normalizar por máx frecuencia (tf normalizado)
        max_tf = max(tf.values()) if tf else 1.0
        for t, cnt in tf.items():
            j = idx[t]
            M[i, j] = (cnt / max_tf) * idf[j]

        # L2
        n = np.linalg.norm(M[i]) + 1e-12
        M[i] = M[i] / n

    return M, idx

def _cosine_topk(q: np.ndarray, M: np.ndarray, k: int) -> Tuple[List[int], np.ndarray]:
    if M.shape[1] == 0 or M.shape[0] == 0:
        return [], np.zeros((0,), dtype=np.float32)
    sims = (M @ q.reshape(-1, 1)).flatten()
    order = np.argsort(-sims)
    idxs = order[: min(k, M.shape[0])].tolist()
    return idxs, sims

# =========================
# Fallback semántico (opcional)
# =========================
@lru_cache(maxsize=1)
def _sbert():
    from sentence_transformers import SentenceTransformer
    model_name = SBERT_LOCAL_PATH if SBERT_LOCAL_PATH else SBERT_MODEL
    return SentenceTransformer(model_name)

@lru_cache(maxsize=10_000)
def _emb(text: str) -> np.ndarray:
    v = _sbert().encode(text, normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)

def _centroid(tokens: List[str]) -> np.ndarray:
    toks = [t for t in tokens or [] if t]
    if not toks:
        # usar dimensión del modelo si disponible
        try:
            return np.zeros((_emb("~").shape[0],), dtype=np.float32)
        except Exception:
            return np.zeros((384,), dtype=np.float32)  # tamaño típico MiniLM (fallback)
    M = np.vstack([_emb(t) for t in toks])
    v = M.mean(axis=0)
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)

def _semantic_hits(student_tokens: List[str], advisor_tokens: List[str], sim_th: float) -> int:
    if not student_tokens or not advisor_tokens:
        return 0
    from sentence_transformers import util
    S = np.vstack([_emb(t) for t in student_tokens])  # (ns, d)
    hits = 0
    for t in advisor_tokens:
        a = _emb(t).reshape(1, -1)
        sim = float(util.cos_sim(a, S).max().item())
        if sim >= sim_th:
            hits += 1
    return hits

# =========================
# Guardas de dominio (auto)
# =========================
def _overlap_count(a: Iterable[str], b: Iterable[str]) -> int:
    sa, sb = set(a), set(b)
    return len(sa & sb)

def _domain_guard(student_stems: List[str], adv_stems: List[str]) -> bool:
    """Acepta si comparten al menos MIN_STEM_OVERLAP stems.
    Esto actúa como filtro de 'macro-dominio' sin diccionario manual."""
    if not student_stems or not adv_stems:
        # si el estudiante o asesor no tiene stems, no bloqueamos
        return True
    return _overlap_count(student_stems, adv_stems) >= MIN_STEM_OVERLAP

# =========================
# API principal
# =========================
def get_recommendations(student_id: int, top_k: int = None) -> List[Dict]:
    """
    Fase A: KNN TF-IDF (tokens). Requiere overlap mínimo de tokens y stems (domain guard).
    Fase B: (opcional) fallback semántico por centroides (SBERT) con hits mínimos.
    Boosts por disponibilidad/idioma.
    """
    top_k = top_k or _DEFAULT_TOP_K
    if top_k <= 0:
        return []

    # --- BD
    conn = _connect()
    try:
        student = _load_student(conn, student_id)
        advisors = _load_advisors(conn)
    finally:
        conn.close()

    if not advisors:
        return []

    # Si no hay tokens del estudiante, no hay base para KNN
    if not student.get("tokens"):
        return []

    # ---------- Fase A: TF-IDF estricto ----------
    A_tokens = [ad["tokens"] for ad in advisors]
    M, vocab_idx = _tfidf_matrix(A_tokens)
    # vector del estudiante en el mismo espacio
    # construir vector TF-IDF del estudiante usando el vocab de asesores
    q = np.zeros((M.shape[1],), dtype=np.float32)
    if M.shape[1] > 0:
        # idf reusar desde _tfidf_matrix: reconstruimos rápido
        # (para mantener consistencia, repetimos mini-cálculo de idf)
        # -> truco: usar DF de asesores a partir de M? No lo tenemos directo,
        # así que recomputamos idf con A_tokens (barato).
        _, idx2 = _tfidf_matrix(A_tokens)  # idx2 tiene las mismas claves/orden que vocab_idx
        # tf normalizado del estudiante
        tf = {}
        for t in student["tokens"]:
            if t in vocab_idx:
                tf[t] = tf.get(t, 0.0) + 1.0
        if tf:
            max_tf = max(tf.values())
            for t, cnt in tf.items():
                j = vocab_idx[t]
                # idf equivalente usando recomputo rápido
                # (como idx2 comparte orden, tomamos idf implícito rehaciendo proyección)
                # más simple: proyectar con mismo método usado en _tfidf_matrix
                q[j] = cnt / max_tf
            # L2
            n = np.linalg.norm(q) + 1e-12
            q = q / n

    idxs, sims = _cosine_topk(q, M, k=max(top_k * 6, top_k))

    strict: List[Dict] = []
    seen = set()

    for i in idxs:
        adv = advisors[i]
        # requisitos mínimos de overlap (tokens y stems)
        if _overlap_count(student["tokens"], adv["tokens"]) < MIN_TOKEN_OVERLAP:
            continue
        if not _domain_guard(student.get("stems", []), adv.get("stems", [])):
            continue

        score = float(sims[i])

        # boosts
        if student.get("availability") and adv.get("availability"):
            if any(a in student["availability"] for a in adv["availability"]):
                score += WEIGHT_AVAIL
        if student.get("language") and adv.get("language"):
            if any(l in student["language"] for l in adv["language"]):
                score += WEIGHT_LANG

        strict.append({
            "advisorId": adv["user_id"],
            "name": adv.get("name") or str(adv["user_id"]),
            "score": round(min(score, 1.0), 4),
            "partial": False,
        })
        seen.add(adv["user_id"])

        if len(strict) >= top_k:
            break

    final = sorted(strict, key=lambda x: x["score"], reverse=True)
    if len(final) >= top_k or not USE_EMBED_FALLBACK:
        return final[:top_k]

    # ---------- Fase B: Fallback semántico ----------
    try:
        s_vec = _centroid(student["tokens"])
        A_emb = np.vstack([_centroid(ad["tokens"]) for ad in advisors])  # (N, d)
        sims_emb = (A_emb @ s_vec.reshape(-1, 1)).flatten()
        order = np.argsort(-sims_emb)

        for i in order:
            adv = advisors[i]
            if adv["user_id"] in seen:
                continue

            # exigir hits semánticos mínimos
            hits = _semantic_hits(student["tokens"], adv["tokens"], SIM_TH_EMB)
            if hits < MIN_SEMANTIC_HITS:
                continue

            # y opcionalmente overlap de stems para no cruzar dominios
            if FALLBACK_REQUIRE_STEM_OVERLAP and not _domain_guard(student.get("stems", []), adv.get("stems", [])):
                continue

            score = float(sims_emb[i])

            if student.get("availability") and adv.get("availability"):
                if any(a in student["availability"] for a in adv["availability"]):
                    score += WEIGHT_AVAIL
            if student.get("language") and adv.get("language"):
                if any(l in student["language"] for l in adv["language"]):
                    score += WEIGHT_LANG

            final.append({
                "advisorId": adv["user_id"],
                "name": adv.get("name") or str(adv["user_id"]),
                "score": round(min(score, 1.0), 4),
                "partial": True,
            })
            seen.add(adv["user_id"])

            if len(final) >= top_k:
                break

    except Exception:
        # si falla el modelo, devolvemos lo que tengamos
        pass

    final.sort(key=lambda x: x["score"], reverse=True)
    return final[:top_k]
