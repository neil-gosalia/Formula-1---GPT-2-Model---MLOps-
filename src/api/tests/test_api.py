from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../api'))

from main import app
client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()['status'] == "ok"

def test_generate_endpoint_exists(): #checks if endpoint exists and responds
    response = client.post("/generate", json={
        "prompt":"Lewis Hamilton",
        "max_length":50,
        "temperature": 0.8,
        "top_p":0.95
    })