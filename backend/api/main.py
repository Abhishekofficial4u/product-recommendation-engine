"""
backend/api/main.py
====================
FastAPI application — the production API for the recommendation engine.

Endpoints:
  GET  /health                      → system status, loaded models
  GET  /recommend/{user_id}         → top-K personalised recommendations
  GET  /similar/{item_id}           → content-similar items
  POST /rate                        → submit a new user rating
  GET  /metrics                     → model evaluation benchmark table
  GET  /items                       → browse item catalogue
  GET  /items/{item_id}             → single item metadata
  GET  /users                       → list of valid user IDs

Run locally:
  cd backend
  uvicorn api.main:app --reload --port 8000

Then open:
  http://localhost:8000/docs   ← Swagger UI (interactive docs)
  http://localhost:8000/redoc  ← ReDoc UI
"""

import os
import sys

# Ensure backend directory is in sys.path for absolute imports
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends, Query, Path, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

security_scheme = HTTPBearer(auto_error=False)

from api.database      import create_tables, get_db, RatingLog, RequestLog, User
from api.schemas       import (RecommendResponse, SimilarResponse, RateRequest,
                                RateResponse, MetricsResponse, ModelMetrics,
                                ItemMetadata, HealthResponse, UserRegister,
                                UserLogin, UserResponse, Token, AnalyticsSummary,
                                ModelUsage, RequestDetail, AgentChatRequest, AgentChatResponse)
from api.model_loader  import (load_all_models, recommend as _recommend,
                                similar_items as _similar, get_metrics,
                                get_health_info, get_all_items, get_item_by_id,
                                get_users)
from api.auth          import (hash_password, verify_password, create_access_token,
                                verify_token)
from api.agent         import recommendation_agent



# ── Startup / shutdown ──────────────────────────────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and create DB tables when the API starts."""
    print("\n=== Starting Recommendation Engine API ===")
    create_tables()
    print("  Database tables created / verified")
    loaded = load_all_models()
    print(f"  Ready - {len(loaded)} models loaded\n")
    yield
    print("=== Shutting down ===")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Product Recommendation Engine API",
    description = """
A hybrid recommendation system built during a 25-day internship.

## Models available
| Model      | Description |
|-----------|-------------|
| `hybrid`  | SVD + Content-based (default, best accuracy) |
| `svd`     | Matrix Factorization |
| `user_cf` | User-based Collaborative Filtering |
| `item_cf` | Item-based Collaborative Filtering |
| `content` | TF-IDF Content-based |
| `popularity` | Popularity baseline |

## Quick start
1. Try `GET /health` to confirm the API is live
2. Use `GET /users` to get valid user IDs
3. Call `GET /recommend/{user_id}` to get personalised recommendations
4. Call `GET /similar/{item_id}` to find similar items
""",
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — allow React frontend (any origin in dev; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],      # Change to ["https://your-frontend.vercel.app"] in prod
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Authentication dependencies & endpoints ──────────────────────────────────

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Dependency to retrieve and verify the user from the JWT Bearer token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    username = verify_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # pyrefly: ignore [bad-argument-type]
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


@app.post(
    "/auth/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Authentication"],
    summary="Register a new user account"
)
async def register(payload: UserRegister, db: Session = Depends(get_db)):
    # Check if username exists
    existing_user = db.query(User).filter(User.username == payload.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    hashed_pwd = hash_password(payload.password)
    db_user = User(
        username=payload.username,
        hashed_password=hashed_pwd
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.post(
    "/auth/login",
    response_model=Token,
    tags=["Authentication"],
    summary="Login to obtain JWT access token"
)
async def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    return Token(access_token=access_token, token_type="bearer")



# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 1 — Health check
# ═══════════════════════════════════════════════════════════════════════════════
@app.get(
    "/health",
    response_model = HealthResponse,
    tags           = ["System"],
    summary        = "Health check — confirm API is live and models are loaded"
)
async def health():
    info = get_health_info()
    return HealthResponse(
        status        = "ok",
        models_loaded = info["models_loaded"],
        total_users   = info["total_users"],
        total_items   = info["total_items"],
        total_ratings = info["total_ratings"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 2 — Personalised recommendations
# ═══════════════════════════════════════════════════════════════════════════════
@app.get(
    "/recommend/{user_id}",
    response_model = RecommendResponse,
    tags           = ["Recommendations"],
    summary        = "Get top-K personalised recommendations for a user",
)
async def recommend(
    user_id: int = Path(..., ge=1, description="User ID"),
    top_k:   int = Query(10,  ge=1, le=50, description="Number of recommendations"),
    model:   str = Query("hybrid", description="Model: hybrid | svd | user_cf | item_cf | content | popularity"),
    db: Session  = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    t_start = time.time()

    VALID_MODELS = {"hybrid","svd","user_cf","item_cf","content","popularity"}
    if model not in VALID_MODELS:
        raise HTTPException(400, f"Invalid model '{model}'. Choose from: {VALID_MODELS}")

    result      = _recommend(user_id, top_k=top_k, model=model)
    elapsed_ms  = round((time.time() - t_start) * 1000, 1)

    # Log request to DB
    try:
        db.add(RequestLog(user_id=user_id, model_used=model,
                          top_k=top_k, response_ms=elapsed_ms))
        db.commit()
    except Exception:
        pass   # don't fail the response if logging fails

    return RecommendResponse(
        user_id      = user_id,
        model_used   = model,
        top_k        = top_k,
        cold_start   = result["cold_start"],
        items        = result["items"],
        response_ms  = elapsed_ms,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 3 — Similar items
# ═══════════════════════════════════════════════════════════════════════════════
@app.get(
    "/similar/{item_id}",
    response_model = SimilarResponse,
    tags           = ["Recommendations"],
    summary        = "Find items similar to a given item using content similarity",
)
async def similar(
    item_id: int = Path(..., ge=1, description="Item ID"),
    top_k:   int = Query(5,   ge=1, le=20, description="Number of similar items"),
):
    result = _similar(item_id, top_k=top_k)

    if not result["items"]:
        raise HTTPException(404, f"Item {item_id} not found or no similar items available.")

    return SimilarResponse(
        item_id       = item_id,
        source_title  = result["source_title"],
        source_genres = result["source_genres"],
        top_k         = top_k,
        items         = result["items"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 4 — Submit a rating
# ═══════════════════════════════════════════════════════════════════════════════
@app.post(
    "/rate",
    response_model = RateResponse,
    tags           = ["Feedback"],
    summary        = "Submit a new user rating (1–5 stars) for an item",
)
async def rate(
    body: RateRequest,
    db:   Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db.add(RatingLog(user_id=body.user_id, item_id=body.item_id, rating=body.rating))
    db.commit()
    return RateResponse(
        message = "Rating saved successfully. Thank you for your feedback!",
        user_id = body.user_id,
        item_id = body.item_id,
        rating  = body.rating,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 5 — Model metrics
# ═══════════════════════════════════════════════════════════════════════════════
@app.get(
    "/metrics",
    response_model = MetricsResponse,
    tags           = ["Analytics"],
    summary        = "Model evaluation metrics — RMSE, Precision@10, Recall@10, NDCG@10",
)
async def metrics():
    data = get_metrics()
    if not data:
        raise HTTPException(503, "Benchmark data not loaded.")
    return MetricsResponse(models=[ModelMetrics(**m) for m in data])


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 6 — Item catalogue
# ═══════════════════════════════════════════════════════════════════════════════
@app.get(
    "/items",
    response_model = List[ItemMetadata],
    tags           = ["Catalogue"],
    summary        = "Browse a sample of the item catalogue",
)
async def items(limit: int = Query(50, ge=1, le=200)):
    return get_all_items(limit=limit)


@app.get(
    "/items/{item_id}",
    response_model = ItemMetadata,
    tags           = ["Catalogue"],
    summary        = "Get metadata for a single item",
)
async def item_detail(item_id: int):
    item = get_item_by_id(item_id)
    if item is None:
        raise HTTPException(404, f"Item {item_id} not found.")
    return item


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 7 — Users list
# ═══════════════════════════════════════════════════════════════════════════════
@app.get(
    "/users",
    response_model = List[int],
    tags           = ["Catalogue"],
    summary        = "Get list of all valid user IDs",
)
async def users():
    return get_users()


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 8 — Rating history (for analytics)
# ═══════════════════════════════════════════════════════════════════════════════
@app.get(
    "/logs/ratings",
    tags    = ["Analytics"],
    summary = "View ratings submitted via POST /rate (newest first)",
)
async def rating_logs(
    limit: int     = Query(20, ge=1, le=100),
    db:    Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    rows = (db.query(RatingLog)
              .order_by(RatingLog.created_at.desc())
              .limit(limit).all())
    return [
        {"id": r.id, "user_id": r.user_id, "item_id": r.item_id,
         "rating": r.rating, "created_at": str(r.created_at)}
        for r in rows
    ]


@app.get(
    "/analytics/summary",
    response_model = AnalyticsSummary,
    tags           = ["Analytics"],
    summary        = "Get aggregated telemetry logs summary for dashboard",
)
async def analytics_summary(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    from sqlalchemy import func

    # 1. Total requests
    total_reqs = db.query(RequestLog).count()

    # 2. Avg response latency
    avg_latency = db.query(func.avg(RequestLog.response_ms)).scalar()
    if avg_latency is not None:
        avg_latency = round(float(avg_latency), 2)

    # 3. Model usage distributions
    usage_rows = (db.query(RequestLog.model_used, func.count(RequestLog.id))
                    .group_by(RequestLog.model_used)
                    .all())
    model_usage = [ModelUsage(model_used=row[0], count=row[1]) for row in usage_rows]

    # 4. Recent request details
    recent_rows = (db.query(RequestLog)
                     .order_by(RequestLog.created_at.desc())
                     .limit(10).all())
    recent_requests = [
        RequestDetail(
            id=r.id,
            user_id=r.user_id,
            model_used=r.model_used,
            top_k=r.top_k,
            response_ms=r.response_ms,
            created_at=r.created_at
        ) for r in recent_rows
    ]

    # 5. Total ratings
    total_ratings = db.query(RatingLog).count()

    return AnalyticsSummary(
        total_requests=total_reqs,
        avg_response_ms=avg_latency,
        model_usage=model_usage,
        recent_requests=recent_requests,
        total_ratings=total_ratings
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINT 9 — Agentic AI Copilot Chat & Tool Execution
# ═══════════════════════════════════════════════════════════════════════════════
@app.post(
    "/agent/chat",
    response_model = AgentChatResponse,
    tags           = ["Agentic AI"],
    summary        = "Conversational AI Copilot with autonomous Tool Calling & RAG Retrieval",
)
async def agent_chat(body: AgentChatRequest):
    """
    Executes an autonomous Agentic AI pipeline.
    Parses user query, determines tool call chain (SVD/Hybrid scoring, similarity, RAG metadata),
    and returns a strictly-typed structured response with execution telemetry.
    """
    return recommendation_agent.process_query(
        query=body.user_query,
        user_id=body.user_id or 1,
        top_k=body.top_k or 5,
        model_preference=body.model_preference or "hybrid"
    )


# ── Run directly for testing ──────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    import uvicorn
    # Add parent directory of api/ to sys.path so 'api' package is discoverable
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
