import pytest
import requests


# from fastapi import TestClient


# test_app: TestClient = TestClient(app)


@pytest.mark.unit
def test_health_check() -> None:
    """Test health endpoint logic without HTTP request"""
    pass


def test_health_endpoint(base_url: str) -> None:
    """Test health endpoint via HTTP request."""
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
