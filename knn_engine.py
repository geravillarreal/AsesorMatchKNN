import pymysql
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ───────────────────────────────────────────────
# CONFIGURACIÓN
# ───────────────────────────────────────────────
DB_CONFIG = dict(
    host="localhost",
    user="root",
    password="admin123",
    db="asesorapp",
    cursorclass=pymysql.cursors.DictCursor
)

# Pesos (suma 1.0)  ← ajusta según tu criterio
WEIGHTS = {
    "areas":        0.30,
    "interests":    0.25,
    "availability": 0.10,
    "modality":     0.10,
    "level":        0.05,
    "language":     0.05,
    "books":        0.15
}

TOP_K_DEFAULT = 5   # resultados a devolver


# ───────────────────────────────────────────────
# UTILIDADES DB
# ───────────────────────────────────────────────
def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


# ───────────────────────────────────────────────
# CARGA DE PERFIL (robusta: devuelve None si falta)
# ───────────────────────────────────────────────
def load_profile(cursor, user_id: int) -> dict | None:
    cursor.execute("SELECT * FROM profile WHERE user_id = %s", user_id)
    p = cursor.fetchone()
    if not p:
        print(f"[WARN] Perfil NO encontrado para user_id={user_id}")
        return None

    # ElementCollections
    cursor.execute("SELECT areas FROM profile_areas WHERE profile_id = %s", p["id"])
    p["areas"] = [r["areas"] for r in cursor.fetchall()]

    cursor.execute("SELECT interests FROM profile_interests WHERE profile_id = %s", p["id"])
    p["interests"] = [r["interests"] for r in cursor.fetchall()]

    cursor.execute("SELECT availability FROM profile_availability WHERE profile_id = %s", p["id"])
    p["availability"] = [r["availability"] for r in cursor.fetchall()]

    # Libros
    cursor.execute("SELECT title FROM book WHERE profile_id = %s", p["id"])
    p["books"] = [r["title"].lower() for r in cursor.fetchall()]

    # Nombre (tabla users)
    cursor.execute("SELECT full_name, faculty FROM users WHERE id = %s", user_id)
    u = cursor.fetchone() or {}
    p["name"] = u.get("full_name", "Sin nombre")
    p["faculty"] = u.get("faculty", "Sin facultad")
    p["user_id"] = user_id
    return _fill_missing(p)


# ───────────────────────────────────────────────
# COMPLETAR ATRIBUTOS FALTANTES
# ───────────────────────────────────────────────
def _fill_missing(profile: dict) -> dict:
    defaults = {
        "areas": [], "interests": [], "availability": [],
        "level": "Desconocido", "modality": "Desconocido",
        "language": "Desconocido", "books": []
    }
    for k, v in defaults.items():
        if k not in profile or profile[k] is None:
            profile[k] = v
    return profile


# ───────────────────────────────────────────────
# ESPACIOS SEMÁNTICOS (vocabularios)
# ───────────────────────────────────────────────
def _build_spaces(profiles: list[dict]) -> dict:
    spaces = {
        "areas":        sorted({a for p in profiles for a in p["areas"]}),
        "interests":    sorted({i for p in profiles for i in p["interests"]}),
        "availability": sorted({av for p in profiles for av in p["availability"]}),
        "levels":       sorted({p["level"] for p in profiles}),
        "modalities":   sorted({p["modality"] for p in profiles}),
        "languages":    sorted({p["language"] for p in profiles}),
        "books": sorted({
            w.lower()
            for p in profiles
            for t in p["books"]
            for w in t.split()
            if len(w) > 3        # omite palabras muy cortas
        })
    }
    # Renombrar plural → singular para simplificar
    return {
        "areas": spaces["areas"],
        "interests": spaces["interests"],
        "availability": spaces["availability"],
        "level": spaces["levels"],
        "modality": spaces["modalities"],
        "language": spaces["languages"],
        "books": spaces["books"]
    }


# ───────────────────────────────────────────────
# VECTORIZACIÓN POR ATRIBUTO
# ───────────────────────────────────────────────
def _vectorize_attr(profile: dict, space: list[str], key: str) -> list[int]:
    if key in ("level", "modality", "language"):
        return [1 if profile[key] == x else 0 for x in space]
    if key == "books":
        text = " ".join(profile["books"])
        return [1 if kw in text else 0 for kw in space]
    # Listas (areas, interests, availability)
    return [1 if x in profile[key] else 0 for x in space]


# ───────────────────────────────────────────────
# SIMILITUD PONDERADA ENTRE DOS PERFILES
# ───────────────────────────────────────────────
def _weighted_similarity(p1: dict, p2: dict, spaces: dict) -> float:
    score = 0.0
    for key, weight in WEIGHTS.items():
        v1 = np.array([_vectorize_attr(p1, spaces[key], key)])
        v2 = np.array([_vectorize_attr(p2, spaces[key], key)])
        # Si ambos vectores son cero → similitud 0
        if not v1.any() or not v2.any():
            continue
        score += weight * cosine_similarity(v1, v2)[0, 0]
    return score


# ───────────────────────────────────────────────
# FUNCIÓN PRINCIPAL: GET RECOMMENDATIONS
# ───────────────────────────────────────────────
def get_recommendations(student_id: int, top_k: int = TOP_K_DEFAULT) -> list[dict]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Estudiante
            student = load_profile(cur, student_id)
            if student is None:
                print(f"[WARN] Estudiante {student_id} sin perfil → retorna lista vacía.")
                return []

            # Traer IDs de asesores
            cur.execute("SELECT id FROM users WHERE role = 'ADVISOR'")
            advisor_ids = [r["id"] for r in cur.fetchall()]

            advisors: list[dict] = []
            for aid in advisor_ids:
                prof = load_profile(cur, aid)
                if prof:
                    advisors.append(prof)

        if not advisors:
            print("[WARN] Sin asesores válidos para comparar.")
            return []

        # Espacios semánticos
        spaces = _build_spaces(advisors + [student])

        # Calcular score ponderado
        results = [
            {
                "advisorId": a["user_id"],
                "name": a["name"],
                "faculty": a["faculty"],
                "score": _weighted_similarity(student, a, spaces)
            }
            for a in advisors
        ]

        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

    finally:
        conn.close()