from flask import request

def log_request() -> None:
    """Simple middleware to log incoming requests."""
    print(f"{request.method} {request.path}")
