"""
backend/api/schemas.py
======================
Pydantic models for request validation and response serialisation.
FastAPI uses these to:
  - Validate incoming JSON automatically
  - Generate the /docs Swagger UI
  - Serialise outgoing JSON responses
"""

from pydantic import BaseModel, Field
from typing   import List, Optional
from datetime import datetime



# ── Response schemas ────────────────────────────────────────────────────────

class RecommendationItem(BaseModel):
    """A single recommended item returned by GET /recommend."""
    item_id:          int
    title:            str
    genres:           str
    predicted_rating: float = Field(..., ge=1.0, le=5.0,
                                    description="Predicted rating on a 1–5 scale")
    rank:             int   = Field(..., description="Position in the ranked list")

    class Config:
        json_schema_extra = {
            "example": {
                "item_id": 318,
                "title": "Item 318",
                "genres": "Drama|Thriller",
                "predicted_rating": 4.21,
                "rank": 1
            }
        }


class RecommendResponse(BaseModel):
    """Full response for GET /recommend/{user_id}."""
    user_id:      int
    model_used:   str
    top_k:        int
    cold_start:   bool = Field(..., description="True if user has fewer than 3 ratings")
    items:        List[RecommendationItem]
    response_ms:  Optional[float] = None

    model_config = {
        "protected_namespaces": ()
    }


class SimilarItem(BaseModel):
    """A single similar item returned by GET /similar/{item_id}."""
    item_id:    int
    title:      str
    genres:     str
    similarity: float = Field(..., ge=0.0, le=1.0)
    rank:       int


class SimilarResponse(BaseModel):
    """Full response for GET /similar/{item_id}."""
    item_id:      int
    source_title: str
    source_genres:str
    top_k:        int
    items:        List[SimilarItem]


class ItemMetadata(BaseModel):
    """Item details returned by GET /items/{item_id}."""
    item_id: int
    title:   str
    genres:  str


# ── Request schemas ─────────────────────────────────────────────────────────

class RateRequest(BaseModel):
    """Body for POST /rate."""
    user_id: int   = Field(..., ge=1, description="User ID")
    item_id: int   = Field(..., ge=1, description="Item ID")
    rating:  float = Field(..., ge=1.0, le=5.0, description="Rating between 1 and 5")

    class Config:
        json_schema_extra = {
            "example": {"user_id": 42, "item_id": 318, "rating": 4.5}
        }


class RateResponse(BaseModel):
    """Response from POST /rate."""
    message: str
    user_id: int
    item_id: int
    rating:  float


# ── Analytics schemas ───────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    """Metrics for one model, returned by GET /metrics."""
    model:  str
    rmse:   Optional[float]
    p10:    Optional[float]
    r10:    Optional[float]
    ndcg10: Optional[float]


class MetricsResponse(BaseModel):
    """Full response for GET /metrics."""
    models: List[ModelMetrics]


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status:         str
    models_loaded:  List[str]
    total_users:    int
    total_items:    int
    total_ratings:  int


# ── Auth schemas ─────────────────────────────────────────────────────────────

class UserRegister(BaseModel):
    """Request body for registering a new user."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=4, max_length=50, description="Password")


class UserLogin(BaseModel):
    """Request body for logging in."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class UserResponse(BaseModel):
    """User representation in API responses."""
    id: int
    username: str
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Response returned upon successful login."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Decoded access token payload data."""
    username: Optional[str] = None


# ── Telemetry Analytics schemas ──────────────────────────────────────────────

class ModelUsage(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_used: str
    count: int


class RequestDetail(BaseModel):
    model_config = {"protected_namespaces": ()}
    id: int
    user_id: int
    model_used: str
    top_k: int
    response_ms: Optional[float]
    created_at: datetime



class AnalyticsSummary(BaseModel):
    model_config = {"protected_namespaces": ()}
    total_requests: int
    avg_response_ms: Optional[float]
    model_usage: List[ModelUsage]
    recent_requests: List[RequestDetail]
    total_ratings: int


# ── Agentic AI & RAG Schemas ──────────────────────────────────────────────────

class ToolCallRecord(BaseModel):
    """Record of a tool call executed by the AI Agent."""
    tool_name: str
    arguments: dict
    output_summary: str


class AgentChatRequest(BaseModel):
    """Request payload for POST /agent/chat."""
    user_query: str = Field(..., description="Natural language prompt for recommendation agent")
    user_id: Optional[int] = Field(1, ge=1, description="Target User ID")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Max recommendations to retrieve")
    model_preference: Optional[str] = Field("hybrid", description="Preferred recommendation model")


class AgentChatResponse(BaseModel):
    """Structured response from the Agentic AI Assistant."""
    model_config = {"protected_namespaces": ()}
    response_text: str = Field(..., description="Synthesized agent response in Markdown format")
    user_id: int
    intent_detected: str
    tool_calls: List[ToolCallRecord]
    recommendations: List[dict]
    rag_sources: List[dict]
    execution_time_ms: float




