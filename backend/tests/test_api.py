import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add parent directory of api/ to sys.path so 'api' package is discoverable
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from api.main import app

@pytest.fixture(scope="module")
def client():
    # Use TestClient as context manager to trigger startup lifespan events (loading models)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_headers(client):
    # Register and login a user for testing protected endpoints
    username = "test_user_for_endpoints"
    password = "testpassword123"
    client.post("/auth/register", json={"username": username, "password": password})
    res = client.post("/auth/login", json={"username": username, "password": password})
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data
    assert "total_users" in data
    assert "total_items" in data
    assert "total_ratings" in data

def test_users(client):
    response = client.get("/users")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert all(isinstance(uid, int) for uid in data)

def test_recommend(client, auth_headers):
    # Test hybrid model (default)
    response = client.get("/recommend/1?top_k=5", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 1
    assert len(data["items"]) == 5
    assert data["model_used"] == "hybrid"

    # Test SVD model
    response = client.get("/recommend/1?top_k=3&model=svd", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["model_used"] == "svd"
    assert len(data["items"]) == 3

    # Test invalid model
    response = client.get("/recommend/1?model=invalid_model", headers=auth_headers)
    assert response.status_code == 400

    # Test invalid user_id validation
    response = client.get("/recommend/0", headers=auth_headers)
    assert response.status_code == 422  # Validation Error

    # Test unauthenticated access (fails with 401)
    response = client.get("/recommend/1?top_k=5")
    assert response.status_code == 401


def test_similar(client):
    response = client.get("/similar/1?top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["item_id"] == 1
    assert len(data["items"]) == 3
    assert "source_title" in data

    # Test invalid item_id validation
    response = client.get("/similar/0")
    assert response.status_code == 422

    # Test not found item_id
    response = client.get("/similar/999999")
    assert response.status_code == 404

def test_items(client):
    response = client.get("/items?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 10

def test_item_detail(client):
    response = client.get("/items/1")
    assert response.status_code == 200
    data = response.json()
    assert data["item_id"] == 1
    assert "title" in data
    assert "genres" in data

def test_rate_and_logs(client, auth_headers):
    # Submit rating
    rate_data = {"user_id": 1, "item_id": 50, "rating": 4.5}
    response = client.post("/rate", json=rate_data, headers=auth_headers)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["user_id"] == 1
    assert res_data["item_id"] == 50
    assert res_data["rating"] == 4.5

    # Check logs
    response = client.get("/logs/ratings?limit=5", headers=auth_headers)
    assert response.status_code == 200
    logs = response.json()
    assert isinstance(logs, list)
    assert len(logs) > 0
    # The newest rating should be the one we just added
    newest = logs[0]
    assert newest["user_id"] == 1
    assert newest["item_id"] == 50
    assert newest["rating"] == 4.5

    # Check unauthenticated access
    response = client.get("/logs/ratings?limit=5")
    assert response.status_code == 401



def test_auth_flow(client):
    import uuid
    rand_username = f"user_{uuid.uuid4().hex[:8]}"
    # Register test user
    reg_data = {"username": rand_username, "password": "securepassword"}
    response = client.post("/auth/register", json=reg_data)
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == rand_username
    assert "id" in data

    # Try registering again (should fail)
    response = client.post("/auth/register", json=reg_data)
    assert response.status_code == 400

    # Login
    login_data = {"username": rand_username, "password": "securepassword"}
    response = client.post("/auth/login", json=login_data)
    assert response.status_code == 200
    token_data = response.json()
    assert token_data["token_type"] == "bearer"
    assert "access_token" in token_data

    # Login with wrong password (should fail)
    login_data = {"username": rand_username, "password": "wrongpassword"}
    response = client.post("/auth/login", json=login_data)
    assert response.status_code == 401


def test_analytics_summary(client, auth_headers):
    # Retrieve analytics summary (authenticated)
    response = client.get("/analytics/summary", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "avg_response_ms" in data
    assert "model_usage" in data
    assert "recent_requests" in data
    assert "total_ratings" in data

    # Retrieve analytics summary (unauthenticated - should fail)
    response = client.get("/analytics/summary")
    assert response.status_code == 401


def test_agent_chat(client):
    # Test agent chat endpoint
    payload = {
        "user_query": "Recommend sci-fi thrillers similar to Star Wars for user 1",
        "user_id": 1,
        "top_k": 3,
        "model_preference": "hybrid"
    }
    response = client.post("/agent/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "response_text" in data
    assert data["user_id"] == 1
    assert len(data["tool_calls"]) > 0
    assert "execution_time_ms" in data




