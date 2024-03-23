import requests


def test_health_endpoint(base_url: str) -> None:
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
