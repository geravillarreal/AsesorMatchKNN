from typing import Any, Dict
from werkzeug.exceptions import BadRequest

def validate_match_request(data: Dict[str, Any] | None) -> int:
    """Validate request payload for match endpoint."""
    if data is None:
        raise BadRequest("Invalid JSON body")
    student_id = data.get("studentId")
    if student_id is None:
        raise BadRequest("studentId is required")
    try:
        return int(student_id)
    except (TypeError, ValueError):
        raise BadRequest("studentId must be an integer")
