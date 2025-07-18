import flask

def log_request() -> None:
    """Simple middleware to log incoming requests."""
    print(f"{flask.request.method} {flask.request.path}")
