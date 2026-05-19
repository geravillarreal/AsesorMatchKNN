"""knn_engine.py

Motor de recomendación (solo engine) para emparejar estudiantes con asesores.

Requisitos del proyecto:
  - NO cambiar endpoints / rutas / formato de respuesta.
  - Mantener la firma:
        get_recommendations(student_id: int, top_k: int|None=None) -> List[Dict]
  - Mantener la salida EXACTA:
        [{"advisorId": int, "name": str, "score": float}, ...]  con score en [0,1]

Notas (env vars útiles):
  - DB_HOST, DB_USER, DB_PASS, DB_NAME, DB_PORT
  - TOP_K_DEFAULT
  - ADVISOR_CACHE_TTL_SEC, EMBEDDING_CACHE_TTL_SEC, EMBEDDING_CACHE_MAXSIZE
  - USE_SEMANTIC (true/false), SBERT_MODEL, SBERT_LOCAL_PATH
  - BM25_K1, BM25_B
  - CANDIDATES_MULTIPLIER, MIN_CANDIDATES, MAX_CANDIDATES
  - AUTO_WEIGHTING (true/false), AUTO_WEIGHTING_ALPHA (0..1)
  - WEIGHT_BM25, WEIGHT_SEMANTIC, WEIGHT_LANGUAGE, WEIGHT_MODALITY, WEIGHT_LEVEL, WEIGHT_AVAILABILITY

"""

from __future__ import annotations

import math
import os
import re
import threading
import time
import unicodedata
from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


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

TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "5"))

# Cache
ADVISOR_CACHE_TTL_SEC = int(os.getenv("ADVISOR_CACHE_TTL_SEC", os.getenv("ADVISOR_CACHE_TTL", "60")))
EMBEDDING_CACHE_TTL_SEC = int(os.getenv("EMBEDDING_CACHE_TTL_SEC", "3600"))
EMBEDDING_CACHE_MAXSIZE = int(os.getenv("EMBEDDING_CACHE_MAXSIZE", "5000"))

# BM25
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

# Candidate generation
CANDIDATES_MULTIPLIER = int(os.getenv("CANDIDATES_MULTIPLIER", "25"))
MIN_CANDIDATES = int(os.getenv("MIN_CANDIDATES", "50"))
MAX_CANDIDATES = int(os.getenv("MAX_CANDIDATES", "400"))

# Semantic
USE_SEMANTIC = os.getenv("USE_SEMANTIC", "false").strip().lower() == "true"
SBERT_MODEL = os.getenv("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
SBERT_LOCAL_PATH = os.getenv("SBERT_LOCAL_PATH", "").strip()

# Cache interno: disponibilidad del modelo semántico (evita reintentos caros)
_SEMANTIC_READY: Optional[bool] = None
_SEMANTIC_READY_LOCK = threading.RLock()

# Feature fusion weights (base)
_W_BM25 = float(os.getenv("WEIGHT_BM25", "0.45"))
_W_SEM = float(os.getenv("WEIGHT_SEMANTIC", "0.25"))
_W_LANG = float(os.getenv("WEIGHT_LANGUAGE", "0.06"))
_W_MODAL = float(os.getenv("WEIGHT_MODALITY", "0.07"))
_W_LEVEL = float(os.getenv("WEIGHT_LEVEL", "0.05"))
_W_AVAIL = float(os.getenv("WEIGHT_AVAILABILITY", "0.12"))

AUTO_WEIGHTING = os.getenv("AUTO_WEIGHTING", "false").strip().lower() == "true"
AUTO_WEIGHTING_ALPHA = float(os.getenv("AUTO_WEIGHTING_ALPHA", "0.35"))  # mezcla base vs std

# Tokenization knobs
USE_BIGRAMS = os.getenv("USE_BIGRAMS", "true").strip().lower() == "true"
USE_STEMS = os.getenv("USE_STEMS", "true").strip().lower() == "true"


# =========================
# Stopwords ES/EN (sin datasets externos)
# =========================

_STOPWORDS_ES = {
    # Muy comunes en ES
    "a", "acerca", "ademas", "adonde", "ahi", "al", "algo", "algun", "alguna", "algunas", "alguno", "algunos",
    "alla", "alli", "ambos", "ante", "antes", "aquel", "aquella", "aquellas", "aquello", "aquellos", "aqui",
    "arriba", "asi", "atras", "aun", "aunque", "bajo", "bien", "cada", "casi", "como", "con", "contra",
    "cual", "cuales", "cualquier", "cualquiera", "cuando", "cuanto", "de", "debajo", "del", "demas",
    "desde", "donde", "dos", "el", "ella", "ellas", "ello", "ellos", "en", "encima", "entre", "era",
    "erais", "eran", "eras", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba",
    "estabais", "estaban", "estabas", "estad", "estada", "estadas", "estado", "estados", "estais", "estamos",
    "estan", "estando", "estar", "estaremos", "estara", "estaran", "estaras", "estare", "estareis", "estaria",
    "estariais", "estariamos", "estarian", "estarias", "estas", "este", "esto", "estos", "estoy", "estuve",
    "estuvimos", "estuvieron", "estuviese", "estuviesen", "estuvieses", "estuvimos", "estuvo", "ex",
    "fin", "fue", "fuera", "fuerais", "fueran", "fueras", "fueron", "fui", "fuimos", "ha", "habeis",
    "habia", "habiais", "habiamos", "habian", "habias", "haber", "habra", "habran", "habras", "habre",
    "habreis", "habria", "habriais", "habriamos", "habrian", "habrias", "han", "has", "hasta", "hay",
    "incluso", "intenta", "intentais", "intentamos", "intentan", "intentar", "intentas", "ir", "jamas",
    "junto", "la", "las", "le", "les", "lo", "los", "mas", "me", "mi", "mientras", "mio", "mis",
    "mucho", "muchos", "muy", "nada", "ni", "no", "nos", "nosotros", "nuestra", "nuestras", "nuestro",
    "nuestros", "o", "os", "otra", "otras", "otro", "otros", "para", "pero", "poco", "por", "porque",
    "primero", "puede", "pueden", "puedo", "que", "quien", "quienes", "se", "sea", "seais", "seamos",
    "sean", "ser", "sera", "seran", "seras", "sere", "sereis", "seria", "seriais", "seriamos", "serian",
    "serias", "si", "sido", "siempre", "sin", "sobre", "solo", "somos", "son", "soy", "su", "sus",
    "tal", "tambien", "tampoco", "tan", "tanto", "te", "teneis", "tenemos", "tener", "tengo", "ti",
    "tiene", "tienen", "toda", "todas", "todo", "todos", "tras", "tu", "tus", "un", "una", "unas",
    "uno", "unos", "usa", "usais", "usamos", "usan", "usar", "usas", "usted", "ustedes", "va", "vais",
    "valor", "vamos", "van", "varias", "varios", "vaya", "verdad", "vos", "vosotros", "y", "ya",
}

_STOPWORDS_EN = {
    # Muy comunes en EN
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "s", "same", "she", "should", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
}


# =========================
# Normalización y tokenización
# =========================


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s = _strip_accents(s)
    # Mantener letras/números; resto -> espacios
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _init_stopwords() -> Set[str]:
    # Normalizamos stopwords (sin acentos) para empatar con _normalize_text
    out: Set[str] = set()
    for w in _STOPWORDS_ES:
        out.add(_normalize_text(w))
    for w in _STOPWORDS_EN:
        out.add(_normalize_text(w))
    # tokens extra que suelen ser ruido en perfiles
    out.update({"etc", "etcetera", "aprox", "aproximadamente", "tema", "temas", "area", "areas", "interes", "intereses"})
    return out


_STOPWORDS = _init_stopwords()


def _simple_stem(tok: str) -> str:
    """Stemmer muy ligero ES/EN (sin NLTK). Conservador para no romper semántica."""
    t = _normalize_text(tok)
    if len(t) < 4:
        return t

    # inglés
    for suf in ("ing", "edly", "ed", "ies", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[: -len(suf)]
            break

    # español (plural / derivaciones comunes)
    for suf in (
        "amientos",
        "amiento",
        "aciones",
        "acion",
        "ciones",
        "cion",
        "mente",
        "idades",
        "idad",
        "ismos",
        "ismo",
        "istas",
        "ista",
        "es",
        "s",
    ):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            t = t[: -len(suf)]
            break
    return t


def _tokenize(text: str) -> List[str]:
    """Tokeniza + normaliza + elimina stopwords.

    Incluye opcionalmente:
      - bigramas (token_i + '_' + token_{i+1})
      - stems (únicos) para robustez morfológica.
    """

    base = _normalize_text(text)
    if not base:
        return []

    toks = [t for t in base.split(" ") if t and t not in _STOPWORDS and len(t) > 1]
    if not toks:
        return []

    out: List[str] = list(toks)

    if USE_BIGRAMS and len(toks) >= 2:
        # Bigramas únicos (evita inflar tf)
        bigrams = []
        seen = set()
        for i in range(len(toks) - 1):
            bg = f"{toks[i]}_{toks[i+1]}"
            if bg not in seen:
                seen.add(bg)
                bigrams.append(bg)
        out.extend(bigrams)

    if USE_STEMS:
        stems = []
        seen = set()
        for t in toks:
            st = _simple_stem(t)
            if st and st not in seen and st not in _STOPWORDS:
                seen.add(st)
                stems.append(st)
        out.extend(stems)

    return out


def _dedup_preserve(seq: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =========================
# Normalización de atributos (idioma / modalidad / nivel)
# =========================


def _norm_language(lang: Optional[str]) -> Optional[str]:
    s = _normalize_text(lang)
    if not s:
        return None
    # mapeos comunes
    if s in {"es", "esp", "espanol"}:
        return "es"
    if s in {"en", "eng", "ingles"}:
        return "en"
    # fallback: 2 primeras letras si parece código
    if len(s) == 2:
        return s
    return s


def _norm_modality(mod: Optional[str]) -> Optional[str]:
    s = _normalize_text(mod)
    if not s:
        return None
    if s in {"remoto", "remote", "online", "virtual", "distancia", "a_distancia"}:
        return "remote"
    if s in {"hibrido", "hybrid", "mixto"}:
        return "hybrid"
    if s in {"presencial", "onsite", "inperson", "in_person", "campus"}:
        return "onsite"
    return s


def _norm_level(level: Optional[str]) -> Optional[int]:
    s = _normalize_text(level)
    if not s:
        return None
    # ordinal básico: licenciatura < maestría < doctorado
    if s in {"licenciatura", "pregrado", "grado", "bachelor", "undergrad", "undergraduate"}:
        return 1
    if s in {"maestria", "master", "masters", "posgrado", "postgrado"}:
        return 2
    if s in {"doctorado", "phd", "doctor", "doctoral"}:
        return 3
    # si viene como número
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    return None


# =========================
# Disponibilidad: parse flexible a slots + overlap
# =========================


_DAY_MAP = {
    # ES
    "lunes": "mon",
    "lun": "mon",
    "martes": "tue",
    "mar": "tue",
    "miercoles": "wed",
    "mie": "wed",
    "jueves": "thu",
    "jue": "thu",
    "viernes": "fri",
    "vie": "fri",
    "sabado": "sat",
    "sab": "sat",
    "domingo": "sun",
    "dom": "sun",
    # EN
    "monday": "mon",
    "mon": "mon",
    "tuesday": "tue",
    "tue": "tue",
    "wednesday": "wed",
    "wed": "wed",
    "thursday": "thu",
    "thu": "thu",
    "friday": "fri",
    "fri": "fri",
    "saturday": "sat",
    "sat": "sat",
    "sunday": "sun",
    "sun": "sun",
}

_SEG_ALIASES = {
    # ES
    "manana": "morning",
    "tarde": "afternoon",
    "noche": "evening",
    "madrugada": "night",
    # EN
    "morning": "morning",
    "afternoon": "afternoon",
    "evening": "evening",
    "night": "night",
    # AM/PM
    "am": "morning",
    "pm": "afternoon",
}


@dataclass(frozen=True)
class _Slot:
    day: Optional[str]  # None => cualquier día
    seg: str            # morning/afternoon/evening/night


_TIME_RANGE_RE = re.compile(r"(?P<h1>\d{1,2})(?::\d{2})?\s*(?:-|a|to)\s*(?P<h2>\d{1,2})(?::\d{2})?")


def _hour_to_segments(h: int) -> Set[str]:
    # mapeo conservador
    if 5 <= h < 12:
        return {"morning"}
    if 12 <= h < 18:
        return {"afternoon"}
    if 18 <= h < 23:
        return {"evening"}
    return {"night"}


def _parse_availability_items(items: Sequence[str]) -> Set[_Slot]:
    slots: Set[_Slot] = set()
    for raw in items or []:
        s = _normalize_text(raw)
        if not s:
            continue

        # días detectados
        days = {canon for k, canon in _DAY_MAP.items() if re.search(rf"\b{k}\b", s)}
        day_list: List[Optional[str]] = sorted(days) if days else [None]

        # segmentos (mañana/tarde/noche)
        segs = {seg for k, seg in _SEG_ALIASES.items() if re.search(rf"\b{k}\b", s)}

        # rango horario explícito
        m = _TIME_RANGE_RE.search(s)
        if m:
            try:
                h1 = int(m.group("h1"))
                h2 = int(m.group("h2"))
                hours = list(range(h1, h2)) if h2 >= h1 else list(range(h1, 24)) + list(range(0, h2))
                segs_from_hours: Set[str] = set()
                for h in hours:
                    segs_from_hours |= _hour_to_segments(h)
                segs |= segs_from_hours
            except Exception:
                pass

        if not segs:
            # Si no detectamos nada, marcamos como 'unknown' (conservador).
            segs = {"unknown"}

        for d in day_list:
            for seg in segs:
                slots.add(_Slot(d, seg))

    return slots


def _availability_coverage(student_slots: Set[_Slot], advisor_slots: Set[_Slot]) -> float:
    """Cobertura de la disponibilidad del estudiante por el asesor (0..1).

    Un slot coincide si:
      - mismo segmento, y
      - día igual o alguno es wildcard (None).
    """
    if not student_slots or not advisor_slots:
        return 0.0

    matched = 0
    for ss in student_slots:
        ok = False
        for aa in advisor_slots:
            if ss.seg != aa.seg:
                continue
            if ss.day is None or aa.day is None or ss.day == aa.day:
                ok = True
                break
        if ok:
            matched += 1
    return matched / max(1, len(student_slots))


# =========================
# Cache TTL (asesores + embeddings)
# =========================


class _TTLValueCache:
    def __init__(self, ttl_sec: int):
        self.ttl_sec = max(1, int(ttl_sec))
        self._lock = threading.RLock()
        self._value = None
        self._expires_at = 0.0

    def get(self):
        with self._lock:
            if self._value is None:
                return None
            if time.time() >= self._expires_at:
                self._value = None
                self._expires_at = 0.0
                return None
            return self._value

    def set(self, value):
        with self._lock:
            self._value = value
            self._expires_at = time.time() + self.ttl_sec

    def clear(self):
        with self._lock:
            self._value = None
            self._expires_at = 0.0


class _TTLLRUCache:
    def __init__(self, ttl_sec: int, maxsize: int):
        self.ttl_sec = max(1, int(ttl_sec))
        self.maxsize = max(100, int(maxsize))
        self._lock = threading.RLock()
        self._data: "OrderedDict[str, Tuple[float, object]]" = OrderedDict()

    def get(self, key: str):
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            exp, val = item
            if now >= exp:
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            return val

    def set(self, key: str, value):
        now = time.time()
        with self._lock:
            self._data[key] = (now + self.ttl_sec, value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self):
        with self._lock:
            self._data.clear()


_ADVISOR_DATASET_CACHE = _TTLValueCache(ADVISOR_CACHE_TTL_SEC)
_EMB_CACHE = _TTLLRUCache(EMBEDDING_CACHE_TTL_SEC, EMBEDDING_CACHE_MAXSIZE)


def refresh_advisors_cache() -> None:
    """Permite invalidar cache externamente (por ejemplo, al actualizar perfiles)."""
    _ADVISOR_DATASET_CACHE.clear()


# =========================
# BM25 (Okapi) con índice invertido
# =========================


@dataclass
class _BM25Index:
    docs: List[List[str]]
    doc_len: List[int]
    avgdl: float
    idf: Dict[str, float]
    inv: Dict[str, List[Tuple[int, int]]]  # term -> [(doc_idx, tf), ...]
    k1: float = BM25_K1
    b: float = BM25_B

    @classmethod
    def build(cls, docs: List[List[str]], k1: float = BM25_K1, b: float = BM25_B) -> "_BM25Index":
        N = len(docs)
        doc_len = [len(d) for d in docs]
        avgdl = (sum(doc_len) / N) if N else 0.0

        df: Dict[str, int] = {}
        inv: Dict[str, List[Tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            if not d:
                continue
            tf = Counter(d)
            for term, cnt in tf.items():
                inv.setdefault(term, []).append((i, int(cnt)))
            for term in set(tf.keys()):
                df[term] = df.get(term, 0) + 1

        idf: Dict[str, float] = {}
        for term, dfi in df.items():
            idf[term] = math.log(((N - dfi + 0.5) / (dfi + 0.5)) + 1.0)

        return cls(docs=docs, doc_len=doc_len, avgdl=avgdl, idf=idf, inv=inv, k1=k1, b=b)

    def score(self, query_terms: Sequence[str]) -> List[float]:
        N = len(self.docs)
        if N == 0:
            return []
        if not query_terms:
            return [0.0] * N

        scores = [0.0] * N
        q_terms = _dedup_preserve(query_terms)
        for term in q_terms:
            postings = self.inv.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, tf in postings:
                dl = self.doc_len[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
                scores[doc_idx] += idf * (tf * (self.k1 + 1.0)) / (denom + 1e-9)
        return scores


# =========================
# Embeddings (Sentence-Transformers) opcional
# =========================


def _safe_dot(u: Sequence[float], v: Sequence[float]) -> float:
    return float(sum((a * b) for a, b in zip(u, v)))


@lru_cache(maxsize=1)
def _get_sbert_model():
    from sentence_transformers import SentenceTransformer

    model_name = SBERT_LOCAL_PATH if SBERT_LOCAL_PATH else SBERT_MODEL
    return SentenceTransformer(model_name)


def _semantic_available() -> bool:
    # Cachea el resultado para no intentar cargar el modelo en cada request.
    if not USE_SEMANTIC:
        return False
    global _SEMANTIC_READY
    if _SEMANTIC_READY is not None:
        return _SEMANTIC_READY
    with _SEMANTIC_READY_LOCK:
        if _SEMANTIC_READY is not None:
            return _SEMANTIC_READY
        try:
            _ = _get_sbert_model()
            _SEMANTIC_READY = True
        except Exception:
            _SEMANTIC_READY = False
        return _SEMANTIC_READY


def _hash_embedding(text: str, dim: int = 64):
    """Fallback determinístico sin dependencias externas.

    NOTA: No pretende ser “semántico real”; solo evita romper el engine
    cuando SBERT no está disponible (y da un embedding estable para tests).
    """
    import hashlib

    toks = _tokenize(text) or _normalize_text(text).split()
    if not toks:
        return [0.0] * dim

    v = [0.0] * dim
    for t in toks:
        h = hashlib.md5(t.encode("utf-8", errors="ignore")).hexdigest()
        idx = int(h[:8], 16) % dim
        v[idx] += 1.0

    norm = math.sqrt(sum(x * x for x in v)) + 1e-12
    return [x / norm for x in v]


def _emb(text: str):
    """Embedding normalizado (unit).

    - Si SBERT está disponible y USE_SEMANTIC=true -> usa Sentence-Transformers.
    - Si no -> fallback determinístico por hashing (no rompe).
    """
    if not _semantic_available():
        return _hash_embedding(text)

    model_name = SBERT_LOCAL_PATH if SBERT_LOCAL_PATH else SBERT_MODEL
    key = f"{model_name}\n{text.strip()}"
    cached = _EMB_CACHE.get(key)
    if cached is not None:
        return cached

    model = _get_sbert_model()
    vec = model.encode(text, normalize_embeddings=True)
    _EMB_CACHE.set(key, vec)
    return vec


def _cosine_sim_unit(u, v) -> float:
    return _safe_dot(u, v)


# =========================
# DB: carga de perfiles
# =========================


def _connect():
    import pymysql

    return pymysql.connect(cursorclass=pymysql.cursors.DictCursor, **_DB_CFG)


def _fetch_profile_fields(conn, user_id: int) -> Dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                u.id AS user_id,
                u.full_name AS full_name,
                u.role AS role,
                p.id AS profile_id,
                p.language AS language,
                p.modality AS modality,
                p.level AS level
            FROM users u
            LEFT JOIN profile p ON p.user_id = u.id
            WHERE u.id = %s
            """,
            (user_id,),
        )
        row = cur.fetchone()
    return row or {}


def _fetch_profile_lists(conn, profile_id: int) -> Tuple[List[str], List[str], List[str]]:
    areas: List[str] = []
    interests: List[str] = []
    availability: List[str] = []

    with conn.cursor() as cur:
        cur.execute("SELECT areas FROM profile_areas WHERE profile_id=%s", (profile_id,))
        areas = [r.get("areas") or "" for r in cur.fetchall()]

        cur.execute("SELECT interests FROM profile_interests WHERE profile_id=%s", (profile_id,))
        interests = [r.get("interests") or "" for r in cur.fetchall()]

        cur.execute("SELECT availability FROM profile_availability WHERE profile_id=%s", (profile_id,))
        availability = [r.get("availability") or "" for r in cur.fetchall()]

    return areas, interests, availability


def _build_topics_text(areas: Sequence[str], interests: Sequence[str]) -> str:
    parts = [p.strip() for p in list(areas or []) + list(interests or []) if p and str(p).strip()]
    seen = set()
    out = []
    for p in parts:
        key = p.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(p.strip())
    return ". ".join(out)


def _load_student(conn, student_id: int) -> Dict:
    row = _fetch_profile_fields(conn, student_id)
    if not row:
        raise ValueError(f"Student {student_id} not found")

    profile_id = row.get("profile_id")
    areas: List[str] = []
    interests: List[str] = []
    availability: List[str] = []
    if profile_id:
        areas, interests, availability = _fetch_profile_lists(conn, int(profile_id))

    topics_text = _build_topics_text(areas, interests)
    tokens = _tokenize(topics_text)

    return {
        "user_id": int(row.get("user_id") or student_id),
        "name": row.get("full_name") or f"Student {student_id}",
        "role": row.get("role"),
        "language": _norm_language(row.get("language")),
        "modality": _norm_modality(row.get("modality")),
        "level": _norm_level(row.get("level")),
        "topics_text": topics_text,
        "tokens": tokens,
        "availability_raw": [a for a in availability if a],
        "availability_slots": _parse_availability_items([a for a in availability if a]),
    }


def _load_advisors_dataset(conn) -> Dict:
    """Carga asesores + construye índice BM25. Esta estructura se cachea (TTL)."""
    advisors: List[Dict] = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                u.id AS user_id,
                u.full_name AS full_name,
                p.id AS profile_id,
                p.language AS language,
                p.modality AS modality,
                p.level AS level
            FROM users u
            LEFT JOIN profile p ON p.user_id = u.id
            WHERE u.role = 'ADVISOR'
            """
        )
        rows = cur.fetchall() or []

        by_pid_areas: Dict[int, List[str]] = {}
        by_pid_interests: Dict[int, List[str]] = {}
        by_pid_avail: Dict[int, List[str]] = {}

        # Prefetch *solo* para perfiles de asesores (evita escanear tablas completas)
        pids = [r.get("profile_id") for r in rows if r.get("profile_id") is not None]
        pids = [int(p) for p in pids if str(p).isdigit()]
        pids = sorted(set(pids))

        def _fetch_chunked(table: str, col: str, out_map: Dict[int, List[str]]):
            if not pids:
                return
            chunk_size = 500
            for off in range(0, len(pids), chunk_size):
                chunk = pids[off : off + chunk_size]
                ph = ",".join(["%s"] * len(chunk))
                cur.execute(f"SELECT profile_id, {col} AS v FROM {table} WHERE profile_id IN ({ph})", tuple(chunk))
                for rr in cur.fetchall() or []:
                    pid = rr.get("profile_id")
                    if pid is None:
                        continue
                    out_map.setdefault(int(pid), []).append(rr.get("v") or "")

        _fetch_chunked("profile_areas", "areas", by_pid_areas)
        _fetch_chunked("profile_interests", "interests", by_pid_interests)
        _fetch_chunked("profile_availability", "availability", by_pid_avail)

    docs_tokens: List[List[str]] = []
    for r in rows:
        uid = int(r.get("user_id"))
        pid = r.get("profile_id")
        pid_i = int(pid) if pid is not None and str(pid).isdigit() else None

        areas = by_pid_areas.get(pid_i, []) if pid_i is not None else []
        interests = by_pid_interests.get(pid_i, []) if pid_i is not None else []
        av = by_pid_avail.get(pid_i, []) if pid_i is not None else []

        topics_text = _build_topics_text(areas, interests)
        tokens = _tokenize(topics_text)

        advisor = {
            "user_id": uid,
            "name": r.get("full_name") or str(uid),
            "language": _norm_language(r.get("language")),
            "modality": _norm_modality(r.get("modality")),
            "level": _norm_level(r.get("level")),
            "topics_text": topics_text,
            "tokens": tokens,
            "availability_raw": [x for x in av if x],
            "availability_slots": _parse_availability_items([x for x in av if x]),
        }
        advisors.append(advisor)
        docs_tokens.append(tokens)

    bm25 = _BM25Index.build(docs_tokens, k1=BM25_K1, b=BM25_B)
    return {"advisors": advisors, "bm25": bm25, "built_at": time.time()}


def _get_advisor_dataset() -> Dict:
    cached = _ADVISOR_DATASET_CACHE.get()
    if cached is not None:
        return cached

    conn = _connect()
    try:
        dataset = _load_advisors_dataset(conn)
    finally:
        conn.close()

    _ADVISOR_DATASET_CACHE.set(dataset)
    return dataset


# =========================
# Features (0..1)
# =========================


def _list_match(a: Sequence[str], b: Sequence[str]) -> bool:
    """True si hay algún elemento en común."""
    sa = {x for x in (a or []) if x}
    sb = {x for x in (b or []) if x}
    return bool(sa & sb)


def _lang_score(student_lang: Optional[str], advisor_lang: Optional[str]) -> float:
    if not student_lang or not advisor_lang:
        return 0.5
    return 1.0 if student_lang == advisor_lang else 0.0


def _modality_score(student_mod: Optional[str], advisor_mod: Optional[str]) -> float:
    if not student_mod or not advisor_mod:
        return 0.5
    if student_mod == advisor_mod:
        return 1.0
    if student_mod == "hybrid" and advisor_mod in {"remote", "onsite"}:
        return 0.7
    if advisor_mod == "hybrid" and student_mod in {"remote", "onsite"}:
        return 0.7
    if {student_mod, advisor_mod} == {"remote", "onsite"}:
        return 0.2
    return 0.0


def _level_score(student_level: Optional[int], advisor_level: Optional[int]) -> float:
    if student_level is None or advisor_level is None:
        return 0.5
    if advisor_level >= student_level:
        return 1.0
    return max(0.0, advisor_level / max(1.0, float(student_level)))


def _availability_score(student_slots: Set[_Slot], advisor_slots: Set[_Slot]) -> float:
    return _availability_coverage(student_slots, advisor_slots)


# =========================
# Auto-weighting (std) + fusión
# =========================


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in weights.items() if v is not None and float(v) >= 0.0}
    s = sum(w.values())
    if s <= 0:
        n = max(1, len(w))
        return {k: 1.0 / n for k in w}
    return {k: v / s for k, v in w.items()}


def _std(vals: List[float]) -> float:
    if not vals:
        return 0.0
    m = sum(vals) / len(vals)
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return math.sqrt(var)


def _auto_weights_std(features: Dict[str, List[float]]) -> Dict[str, float]:
    stds = {k: _std(vs) for k, vs in features.items()}
    if sum(stds.values()) <= 1e-12:
        n = max(1, len(stds))
        return {k: 1.0 / n for k in stds}
    return _normalize_weights(stds)


def _blend_weights(base: Dict[str, float], auto: Dict[str, float], alpha: float) -> Dict[str, float]:
    alpha = min(1.0, max(0.0, float(alpha)))
    keys = set(base) | set(auto)
    mixed = {k: (1.0 - alpha) * base.get(k, 0.0) + alpha * auto.get(k, 0.0) for k in keys}
    return _normalize_weights(mixed)


# =========================
# Legacy helpers (compatibilidad con tests antiguos)
# =========================


_WEIGHTS: Dict[str, float] = {
    "bm25": _W_BM25,
    "semantic": _W_SEM,
    "language": _W_LANG,
    "modality": _W_MODAL,
    "level": _W_LEVEL,
    "availability": _W_AVAIL,
    "topics": _W_BM25,
}


def _coverage(student_topics: Sequence[str], advisor_topics: Sequence[str]) -> float:
    if not advisor_topics or not student_topics:
        return 0.0
    s_terms = [str(t) for t in student_topics if t is not None]
    a_terms = [str(t) for t in advisor_topics if t is not None]
    if not s_terms or not a_terms:
        return 0.0
    s_vecs = [_emb(t) for t in s_terms]
    covs: List[float] = []
    for t in a_terms:
        av = _emb(t)
        best = -1.0
        for sv in s_vecs:
            sim = _cosine_sim_unit(av, sv)
            if sim > best:
                best = sim
        covs.append(max(0.0, float(best)))
    return float(sum(covs) / len(covs))


def _score(student: Dict, advisor: Dict) -> float:
    topics_score = _coverage(student.get("topics", []), advisor.get("topics", []))
    avail_score = 1.0 if _list_match(student.get("availability", []), advisor.get("availability", [])) else 0.0
    lang_score = 1.0 if _list_match(student.get("language", []), advisor.get("language", [])) else 0.0
    return (
        topics_score * _WEIGHTS["topics"]
        + avail_score * _WEIGHTS["availability"]
        + lang_score * _WEIGHTS["language"]
    )


# =========================
# API principal
# =========================


def get_recommendations(student_id: int, top_k: int | None = None) -> List[Dict]:
    k = int(top_k or TOP_K_DEFAULT)
    if k <= 0:
        return []

    conn = _connect()
    try:
        student = _load_student(conn, int(student_id))
    finally:
        conn.close()

    dataset = _get_advisor_dataset()
    advisors: List[Dict] = dataset.get("advisors", [])
    bm25: _BM25Index = dataset.get("bm25")
    if not advisors:
        return []

    query_tokens = student.get("tokens", []) or []
    bm25_scores = bm25.score(query_tokens)
    max_bm25 = max(bm25_scores) if bm25_scores else 0.0

    cand_n = min(len(advisors), max(MIN_CANDIDATES, k * max(1, CANDIDATES_MULTIPLIER)))
    cand_n = min(cand_n, max(1, MAX_CANDIDATES))

    if not query_tokens or max_bm25 <= 1e-12:
        if student.get("language"):
            pref = [i for i, a in enumerate(advisors) if _lang_score(student.get("language"), a.get("language")) > 0.0]
            candidate_idxs = pref if pref else list(range(len(advisors)))
        else:
            candidate_idxs = list(range(len(advisors)))
        candidate_idxs = candidate_idxs[:cand_n]
    else:
        import heapq

        top = heapq.nlargest(cand_n, enumerate(bm25_scores), key=lambda x: x[1])
        candidate_idxs = [i for i, _ in top]

    cand_bm25_raw = [bm25_scores[i] if i < len(bm25_scores) else 0.0 for i in candidate_idxs]
    bm25_den = max(cand_bm25_raw) if cand_bm25_raw else 0.0
    bm25_feat = [(s / bm25_den) if bm25_den > 1e-12 else 0.0 for s in cand_bm25_raw]

    lang_feat = [_lang_score(student.get("language"), advisors[i].get("language")) for i in candidate_idxs]
    modal_feat = [_modality_score(student.get("modality"), advisors[i].get("modality")) for i in candidate_idxs]
    level_feat = [_level_score(student.get("level"), advisors[i].get("level")) for i in candidate_idxs]
    avail_feat = [
        _availability_score(student.get("availability_slots", set()), advisors[i].get("availability_slots", set()))
        for i in candidate_idxs
    ]

    sem_enabled = _semantic_available() and bool(student.get("topics_text"))
    sem_raw: List[float] = [0.0 for _ in candidate_idxs]
    if sem_enabled:
        try:
            s_vec = _emb(student.get("topics_text") or "")
            for j, i in enumerate(candidate_idxs):
                adv_text = advisors[i].get("topics_text") or ""
                if not adv_text:
                    sem_raw[j] = 0.0
                    continue
                a_vec = _emb(adv_text)
                sem_raw[j] = max(0.0, _cosine_sim_unit(s_vec, a_vec))
        except Exception:
            sem_enabled = False
            sem_raw = [0.0 for _ in candidate_idxs]

    sem_den = max(sem_raw) if sem_raw else 0.0
    sem_feat = [(s / sem_den) if sem_den > 1e-12 else 0.0 for s in sem_raw]

    feature_vectors: Dict[str, List[float]] = {
        "language": lang_feat,
        "modality": modal_feat,
        "level": level_feat,
        "availability": avail_feat,
    }
    base_weights: Dict[str, float] = {
        "language": _W_LANG,
        "modality": _W_MODAL,
        "level": _W_LEVEL,
        "availability": _W_AVAIL,
    }

    if query_tokens:
        feature_vectors["bm25"] = bm25_feat
        base_weights["bm25"] = _W_BM25

    if sem_enabled:
        feature_vectors["semantic"] = sem_feat
        base_weights["semantic"] = _W_SEM

    base_w = _normalize_weights(base_weights)
    if AUTO_WEIGHTING:
        auto_w = _auto_weights_std(feature_vectors)
        weights = _blend_weights(base_w, auto_w, AUTO_WEIGHTING_ALPHA)
    else:
        weights = base_w

    results: List[Dict] = []
    for pos, adv_idx in enumerate(candidate_idxs):
        adv = advisors[adv_idx]
        score = 0.0
        for fname, w in weights.items():
            fv = feature_vectors.get(fname)
            if fv is None:
                continue
            score += float(fv[pos]) * float(w)
        score = max(0.0, min(1.0, score))
        results.append({
            "advisorId": int(adv.get("user_id")),
            "name": str(adv.get("name") or adv.get("user_id")),
            "score": float(round(score, 4)),
            "topics_text": str(adv.get("topics_text") or ""),
            "tokens": adv.get("tokens") or [],
        })

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]
