from typing import Any, Dict

from werkzeug.exceptions import BadRequest
from pydantic import BaseModel


class MatchRequest(BaseModel):
    studentId: int


class Recommendation(BaseModel):
    advisorId: int
    name: str
    score: float


def validate_match_request(data: Dict[str, Any] | None) -> MatchRequest:
    """Validate request payload for match endpoint."""
    if data is None:
        raise BadRequest("Invalid JSON body")
    try:
        return MatchRequest(**data)
    except Exception as e:
        raise BadRequest(str(e))
