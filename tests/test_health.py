def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.0"
    assert data["environment"] == "development"


def test_health_check_uses_runtime_settings(monkeypatch, client):
    class DummySettings:
        APP_VERSION = "9.9.9"
        ENVIRONMENT = "staging"

    monkeypatch.setattr("app.routes.health.get_settings", lambda: DummySettings())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["version"] == "9.9.9"
    assert response.json()["environment"] == "staging"
