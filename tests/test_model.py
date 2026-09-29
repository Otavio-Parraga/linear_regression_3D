import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import model as M
from app.main import app


def test_cost_zero_on_perfect_fit():
    assert M.cost([1, 2, 3], [1, 2, 3], 0, 1) == 0.0


def test_cost_hand_computed():
    # h = [1,1,1], residuals = [0,-1,-2], sum sq = 5, J = 5/(2*3)
    assert M.cost([1, 2, 3], [1, 2, 3], 1, 0) == pytest.approx(5 / 6)
    # h = 0, residuals = -y, sum sq = 14, J = 14/6
    assert M.cost([1, 2, 3], [1, 2, 3], 0, 0) == pytest.approx(14 / 6)


def test_predict():
    np.testing.assert_allclose(M.predict([0, 1, 2], 2, 3), [2, 5, 8])


@pytest.mark.parametrize("t0,t1", [(0, 0), (10, 2), (-5, 7.5), (50, 4)])
def test_gradient_matches_finite_differences(t0, t1):
    x, y, eps = M.X, M.Y, 1e-5
    g0, g1 = M.gradient(x, y, t0, t1)
    fd0 = (M.cost(x, y, t0 + eps, t1) - M.cost(x, y, t0 - eps, t1)) / (2 * eps)
    fd1 = (M.cost(x, y, t0, t1 + eps) - M.cost(x, y, t0, t1 - eps)) / (2 * eps)
    assert g0 == pytest.approx(fd0, rel=1e-6, abs=1e-6)
    assert g1 == pytest.approx(fd1, rel=1e-6, abs=1e-6)


def _features():
    z, _, _ = M.normalize(M.X)
    return {"original": M.X, "normalized": z}


@pytest.mark.parametrize("scale", ["original", "normalized"])
def test_gradient_zero_at_theta_star(scale):
    f = _features()[scale]
    t0, t1 = M.normal_equation(f, M.Y)
    g0, g1 = M.gradient(f, M.Y, t0, t1)
    assert abs(g0) < 1e-9 and abs(g1) < 1e-9


def test_normalization_uses_population_std():
    z, mean, std = M.normalize(M.X)
    assert mean == pytest.approx(5.5)
    assert std == pytest.approx(np.sqrt(8.25))
    assert np.mean(z) == pytest.approx(0, abs=1e-12)
    assert np.std(z) == pytest.approx(1)


@pytest.fixture(scope="module")
def payload():
    return M.build_model_payload()


@pytest.mark.parametrize("scale", ["original", "normalized"])
def test_grid_min_near_theta_star(payload, scale):
    s = payload["scales"][scale]
    g = s["grid"]
    J = np.array(g["J"])
    i, j = np.unravel_index(np.argmin(J), J.shape)
    center = M.GRID_N // 2
    assert abs(i - center) <= 1 and abs(j - center) <= 1
    t0s, t1s = s["theta_star"]
    assert g["t0"][center] == pytest.approx(t0s, abs=1e-5)
    assert g["t1"][center] == pytest.approx(t1s, abs=1e-5)


@pytest.mark.parametrize("scale", ["original", "normalized"])
def test_grid_orientation(scale):
    f = _features()[scale]
    t0s, t1s = M.normal_equation(f, M.Y)
    w0, w1 = M.HALF_WIDTH[scale]
    g0 = np.linspace(t0s - w0, t0s + w0, M.GRID_N)
    g1 = np.linspace(t1s - w1, t1s + w1, M.GRID_N)
    J = M.cost_grid(f, M.Y, g0, g1)
    assert J.shape == (M.GRID_N, M.GRID_N)
    for i, j in [(0, 0), (0, 60), (60, 0), (10, 45), (37, 3), (30, 30)]:
        assert J[i, j] == pytest.approx(M.cost(f, M.Y, g0[j], g1[i]))


def test_cost_grid_non_square_orientation():
    t0 = np.array([0.0, 1.0, 2.0])
    t1 = np.array([0.0, 1.0])
    J = M.cost_grid([1, 2, 3], [1, 2, 3], t0, t1)
    assert J.shape == (2, 3)
    assert J[1, 0] == pytest.approx(0.0)  # t0=0, t1=1
    assert J[0, 1] == pytest.approx(5 / 6)  # t0=1, t1=0


@pytest.mark.parametrize("t0,t1", [(40, 5), (43.5, 4.4), (0, 0), (-10, 12)])
def test_original_normalized_equivalence(t0, t1):
    z, mean, std = M.normalize(M.X)
    t0n, t1n = t0 + t1 * mean, t1 * std
    np.testing.assert_allclose(M.predict(M.X, t0, t1), M.predict(z, t0n, t1n))
    assert M.cost(M.X, M.Y, t0, t1) == pytest.approx(M.cost(z, M.Y, t0n, t1n))


def test_theta_star_equivalence_and_same_J():
    z, mean, std = M.normalize(M.X)
    t0, t1 = M.normal_equation(M.X, M.Y)
    t0n, t1n = M.normal_equation(z, M.Y)
    assert t0n == pytest.approx(t0 + t1 * mean)
    assert t1n == pytest.approx(t1 * std)
    assert M.cost(M.X, M.Y, t0, t1) == pytest.approx(M.cost(z, M.Y, t0n, t1n))


def test_api_model():
    client = TestClient(app)
    r = client.get("/api/model")
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"data", "stats", "scales"}
    assert d["data"]["x"] == [float(v) for v in range(1, 11)]
    assert len(d["data"]["y"]) == 10
    assert d["stats"]["mean"] == pytest.approx(5.5)
    assert d["stats"]["std"] == pytest.approx(2.872281, abs=1e-6)
    assert set(d["scales"]) == {"original", "normalized"}
    for name, s in d["scales"].items():
        assert set(s) == {"feature", "theta_star", "J_star", "range", "grid"}
        assert len(s["feature"]) == 10
        assert len(s["theta_star"]) == 2
        assert set(s["range"]) == {"t0", "t1"}
        assert len(s["range"]["t0"]) == 2 and len(s["range"]["t1"]) == 2
        g = s["grid"]
        n = M.GRID_N
        assert len(g["t0"]) == n and len(g["t1"]) == n
        assert len(g["J"]) == n and all(len(row) == n for row in g["J"])
    assert d["scales"]["original"]["feature"] == d["data"]["x"]
    assert d["scales"]["original"]["J_star"] == pytest.approx(
        d["scales"]["normalized"]["J_star"], abs=1e-5
    )


def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_model_payload_is_gzipped():
    client = TestClient(app)
    r = client.get("/api/model", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and r.headers.get("content-encoding") == "gzip"


def test_index_links_versioned_assets_and_asks_to_revalidate():
    from app.main import ASSET_VERSION
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"
    for name in ("style.css", "i18n.js", "app.js"):
        assert f"/static/{name}?v={ASSET_VERSION}" in r.text
    assert client.get(f"/static/app.js?v={ASSET_VERSION}").headers["cache-control"] == "no-cache"
