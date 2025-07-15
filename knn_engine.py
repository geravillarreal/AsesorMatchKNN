import pymysql
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="admin123",
        db="asesorapp",
        cursorclass=pymysql.cursors.DictCursor
    )

def load_profile(cursor, user_id):
    cursor.execute("SELECT * FROM profile WHERE user_id = %s", (user_id,))
    p = cursor.fetchone()
    if not p:
        print(f"[WARN] Profile for user {user_id} not found")
        return None

    cursor.execute("SELECT areas FROM profile_areas WHERE profile_id = %s", (p["id"],))
    p["areas"] = [r["areas"] for r in cursor.fetchall()]

    cursor.execute("SELECT interests FROM profile_interests WHERE profile_id = %s", (p["id"],))
    p["interests"] = [r["interests"] for r in cursor.fetchall()]

    cursor.execute("SELECT availability FROM profile_availability WHERE profile_id = %s", (p["id"],))
    p["availability"] = [r["availability"] for r in cursor.fetchall()]

    cursor.execute("SELECT full_name FROM users WHERE id = %s", (user_id,))
    p["name"] = cursor.fetchone()["full_name"]

    cursor.execute("SELECT title FROM book WHERE profile_id = %s", (p["id"],))
    p["books"] = [r["title"].lower() for r in cursor.fetchall()] 

    print(p)

    return p


def vectorize(profile, areas_set, interests_set, availability_set, levels, modalities, languages, book_keywords):
    v = []
    v += [1 if x in profile["areas"] else 0 for x in areas_set]
    v += [1 if x in profile["interests"] else 0 for x in interests_set]
    v += [1 if x in profile["availability"] else 0 for x in availability_set]
    v += [1 if profile["level"] == x else 0 for x in levels]
    v += [1 if profile["modality"] == x else 0 for x in modalities]
    v += [1 if profile["language"] == x else 0 for x in languages]

    book_string = " ".join(profile.get("books", []))
    v += [1 if kw in book_string else 0 for kw in book_keywords]

    return v

import pymysql
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def get_db_connection():
    return pymysql.connect(
        host="localhost",
        user="root",
        password="admin123",
        db="asesorapp",
        cursorclass=pymysql.cursors.DictCursor
    )

def load_profile(cursor, user_id):
    cursor.execute("SELECT * FROM profile WHERE user_id = %s", (user_id,))
    p = cursor.fetchone()
    if not p:
        print(f"[WARN] Profile for user {user_id} not found")
        return None

    cursor.execute("SELECT areas FROM profile_areas WHERE profile_id = %s", (p["id"],))
    p["areas"] = [r["areas"] for r in cursor.fetchall()]

    cursor.execute("SELECT interests FROM profile_interests WHERE profile_id = %s", (p["id"],))
    p["interests"] = [r["interests"] for r in cursor.fetchall()]

    cursor.execute("SELECT availability FROM profile_availability WHERE profile_id = %s", (p["id"],))
    p["availability"] = [r["availability"] for r in cursor.fetchall()]

    cursor.execute("SELECT full_name FROM users WHERE id = %s", (user_id,))
    result = cursor.fetchone()
    p["name"] = result["full_name"] if result else "Sin nombre"

    cursor.execute("SELECT title FROM book WHERE profile_id = %s", (p["id"],))
    p["books"] = [r["title"].lower() for r in cursor.fetchall()]

    p["user_id"] = user_id  # lo necesitas en el resultado final

    return p

def vectorize(profile, areas_set, interests_set, availability_set, levels, modalities, languages, book_keywords):
    v = []
    v += [1 if x in profile["areas"] else 0 for x in areas_set]
    v += [1 if x in profile["interests"] else 0 for x in interests_set]
    v += [1 if x in profile["availability"] else 0 for x in availability_set]
    v += [1 if profile["level"] == x else 0 for x in levels]
    v += [1 if profile["modality"] == x else 0 for x in modalities]
    v += [1 if profile["language"] == x else 0 for x in languages]

    book_string = " ".join(profile.get("books", []))
    v += [1 if kw in book_string else 0 for kw in book_keywords]

    return v

def get_recommendations(student_id, top_k=5):
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Cargar perfil del estudiante
            student = load_profile(cursor, student_id)
            if student is None:
                print(f"[WARN] No se encontró el perfil del estudiante con ID {student_id}.")
                return []

            # Cargar asesores
            cursor.execute("SELECT id FROM users WHERE role = 'ADVISOR'")
            advisor_ids = [r["id"] for r in cursor.fetchall()]

            advisors = []
            for aid in advisor_ids:
                p = load_profile(cursor, aid)
                if p:
                    advisors.append(p)
                else:
                    print(f"[WARN] Se omitió advisor con ID {aid} por falta de perfil.")

            if not advisors:
                print(f"[WARN] No advisor profiles available to compare.")
                return []

        # Espacios semánticos
        areas = sorted(set(a for p in advisors + [student] for a in p["areas"]))
        interests = sorted(set(i for p in advisors + [student] for i in p["interests"]))
        availability = sorted(set(av for p in advisors + [student] for av in p["availability"]))
        levels = sorted(set(p["level"] for p in advisors + [student]))
        modalities = sorted(set(p["modality"] for p in advisors + [student]))
        languages = sorted(set(p["language"] for p in advisors + [student]))

        book_keywords = sorted(set(
            word.lower()
            for p in advisors + [student]
            for title in p.get("books", [])
            for word in title.split() if len(word) > 3
        ))

        # Vectores
        v_student = np.array([
            vectorize(student, areas, interests, availability, levels, modalities, languages, book_keywords)
        ])
        v_advisors = np.array([
            vectorize(p, areas, interests, availability, levels, modalities, languages, book_keywords)
            for p in advisors
        ])

        if v_advisors.size == 0:
            print("[ERROR] No valid advisor vectors were generated.")
            return []

        sims = cosine_similarity(v_student, v_advisors)[0]
        results = sorted([
            {"advisorId": p["user_id"], "name": p["name"], "score": float(sim)}
            for p, sim in zip(advisors, sims)
        ], key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    finally:
        conn.close()

