from fastapi.testclient import TestClient
from src.main import app  # adjust import if needed

client = TestClient(app)

def test_recommend_random():
    """Test that /recommend/random runs and returns a response"""
    response = client.get("/recommend/random?genre=rock&n=5")
    assert response.status_code == 200
    assert "recommendations" in response.json()

def test_recommend_filtered():
    """Test that /recommend/filtered runs with tempo + exercise_id"""
    payload = {
        "tempo": 120,
        "exercise_id": 1,
        "genre": "rock"
    }
    response = client.post("/recommend/filtered", json=payload)
    assert response.status_code == 200
    assert "recommendations" in response.json()
