
import os
import uuid

import pytest
from fastapi.testclient import TestClient

# Use Docker Compose defaults locally, but preserve URLs provided by CircleCI.
os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://leaderboard:leaderboard@postgres:5432/leaderboard",
)
os.environ.setdefault("REDIS_URL", "redis://redis:6379/0")

from app.main import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def unique_suffix():
    return uuid.uuid4().hex[:12]


def test_health_checks_postgres_and_redis(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["postgres"] is True
    assert response.json()["redis"] is True


def test_submit_score_and_read_player(client):
    suffix = unique_suffix()
    player = f"integration-player-{suffix}"
    run_id = f"integration-run-{suffix}"

    response = client.post(
        "/scores",
        json={
            "player": player,
            "run_id": run_id,
            "region": "north",
            "score": 100,
            "seconds": 20,
        },
    )

    assert response.status_code == 201
    assert response.json()["player"] == player
    assert response.json()["run_id"] == run_id
    assert response.json()["score"] == 100

    player_response = client.get(f"/players/{player}")

    assert player_response.status_code == 200
    player_data = player_response.json()
    assert player_data["player"] == player
    assert player_data["region"] == "north"
    assert player_data["global_rank"] >= 1

    board_response = client.get(
        "/leaderboard",
        params={"scope": "global", "page": 1, "size": 100},
    )

    assert board_response.status_code == 200
    assert board_response.json()["total_players"] >= 1
    assert isinstance(board_response.json()["entries"], list)


def test_duplicate_run_id_is_rejected(client):
    suffix = unique_suffix()
    run_id = f"integration-duplicate-{suffix}"

    payload = {
        "player": f"integration-player-{suffix}",
        "run_id": run_id,
        "region": "south",
        "score": 100,
        "seconds": 20,
    }

    first = client.post("/scores", json=payload)
    second = client.post("/scores", json=payload)

    assert first.status_code == 201
    assert second.status_code == 409


def test_invalid_region_is_rejected(client):
    response = client.post(
        "/scores",
        json={
            "player": f"integration-player-{unique_suffix()}",
            "run_id": f"integration-run-{unique_suffix()}",
            "region": "invalid-region",
            "score": 100,
            "seconds": 20,
        },
    )

    assert response.status_code == 400


def test_admin_rebuild_restores_leaderboard(client):
    response = client.post("/admin/rebuild")

    assert response.status_code == 200
    body = response.json()

    assert body["rebuilt"] is True
    assert "players" in body
    assert "regions" in body
    assert body["max_score"] == 1_000_000
