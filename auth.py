import os
import datetime
from functools import wraps
from typing import Any, Dict

import jwt
import flask
from werkzeug.exceptions import Unauthorized, BadRequest

SECRET_KEY = os.getenv("JWT_SECRET", "secret")


def create_token(user_id: int) -> str:
    payload = {
        "user_id": user_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


def verify_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise Unauthorized("Token expired")
    except jwt.InvalidTokenError:
        raise Unauthorized("Invalid token")


def auth_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth_header = flask.request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise Unauthorized("Missing token")
        token = auth_header.split(" ", 1)[1]
        flask.request.user = verify_token(token)
        return func(*args, **kwargs)

    return wrapper
