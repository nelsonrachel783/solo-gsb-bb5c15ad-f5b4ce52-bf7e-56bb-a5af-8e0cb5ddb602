"""/api/wavefront-evaluations 的大坐标直线拟合边界回归测试。

- 无界面、观察线 x=1e200、中心方向 (1,0)、半角 30°、采样 3 条且 n_start=1：
  交点与光程均为有限值（约 1e200 量级），旧实现直接对偏移离差平方求和
  （约 (5.8e199)²，超出双精度范围）触发 OverflowError，接口返回 500；
  尺度归一化后截距、斜率、逐光线残差与残差 RMS 均为有限值；
- 有效光线不足 3 条时拟合参数与残差仍为 null（既有拟合失效语义不回退），
  但光程与偏移正常返回；
- 普通玻璃板拟合结果与 README 示例数值一致（普通尺度结果保持原有语义）。
"""
from __future__ import annotations

import math

import pytest

WAVEFRONT_PATH = "/api/wavefront-evaluations"


def _large_coordinate_body(sample_count: int = 3) -> dict:
    return {
        "start": {"x": 0.0, "y": 0.0},
        "center_direction": {"x": 1.0, "y": 0.0},
        "half_angle_deg": 30.0,
        "sample_count": sample_count,
        "n_start": 1.0,
        "interfaces": [],
        "observation_line": {
            "point": {"x": 1e200, "y": 0.0},
            "normal": {"x": 1.0, "y": 0.0},
        },
    }


def test_wavefront_large_coordinate_fit_finite(client):
    """观察线 x=1e200、无界面、3 条光线：旧实现 500，现 200 且拟合量有限。"""
    resp = client.post(WAVEFRONT_PATH, json=_large_coordinate_body())
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ray_count"] == 3
    assert data["hit_count"] == 3
    # 截距、斜率、残差 RMS 均为有限值（旧实现离差平方溢出，整组为 null/500）
    assert data["intercept"] == pytest.approx(1.1031336922528344e200)
    assert data["slope"] == 0.0
    assert data["residual_rms"] == pytest.approx(7.292653316078565e198)
    assert math.isfinite(data["intercept"])
    assert math.isfinite(data["residual_rms"])
    for ray in data["rays"]:
        assert ray["status"] == "completed"
        assert ray["optical_path"] is not None and math.isfinite(ray["optical_path"])
        assert ray["offset"] is not None and math.isfinite(ray["offset"])
        assert ray["residual"] is not None and math.isfinite(ray["residual"])
        assert math.isfinite(ray["observation_point"]["x"])
        assert math.isfinite(ray["observation_point"]["y"])
    # 边缘光线（±30°）偏移 ±1e200·tan(30°)、光程 1e200/cos(30°)
    edge = sorted(data["rays"], key=lambda r: abs(r["emission_angle_deg"]), reverse=True)
    assert abs(edge[0]["offset"]) == pytest.approx(1e200 * math.tan(math.radians(30.0)))
    assert edge[0]["optical_path"] == pytest.approx(1e200 / math.cos(math.radians(30.0)))


def test_wavefront_fewer_than_three_rays_fit_null(client):
    """仅 2 条有效光线：拟合参数与逐光线残差为 null，光程/偏移仍正常返回。"""
    resp = client.post(WAVEFRONT_PATH, json=_large_coordinate_body(sample_count=2))
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["hit_count"] == 2
    assert data["intercept"] is None
    assert data["slope"] is None
    assert data["residual_rms"] is None
    for ray in data["rays"]:
        assert ray["status"] == "completed"
        assert ray["optical_path"] is not None
        assert ray["offset"] is not None
        assert ray["residual"] is None


def test_wavefront_identical_offsets_fit_null():
    """偏移全相同（斜率不可定）：拟合返回 None，即接口各拟合字段为 null。"""
    from app.geometry import linear_least_squares

    fit = linear_least_squares([(2.0, 1.0), (2.0, 3.0), (2.0, 7.0)])
    assert fit is None


def test_wavefront_glass_slab_matches_reference(client):
    """普通 1.0 -> 1.5 -> 1.0 玻璃板：拟合结果与 README 示例数值一致。"""
    body = {
        "start": {"x": 0.0, "y": 0.0},
        "center_direction": {"x": 1.0, "y": 0.0},
        "half_angle_deg": 10.0,
        "sample_count": 5,
        "n_start": 1.0,
        "interfaces": [
            {"point": {"x": 2.0, "y": 0.0}, "normal": {"x": 1.0, "y": 0.0}, "n_next": 1.5},
            {"point": {"x": 4.0, "y": 0.0}, "normal": {"x": 1.0, "y": 0.0}, "n_next": 1.0},
        ],
        "observation_line": {
            "point": {"x": 10.0, "y": 0.0},
            "normal": {"x": 1.0, "y": 0.0},
        },
    }
    resp = client.post(WAVEFRONT_PATH, json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["hit_count"] == 5
    assert data["slope"] == 0.0
    assert data["intercept"] == pytest.approx(11.071742133570087, rel=1e-12)
    assert data["residual_rms"] == pytest.approx(0.06019273926764479, rel=1e-12)
    rays = {r["emission_angle_deg"]: r for r in data["rays"]}
    assert rays[-10.0]["residual"] == pytest.approx(0.07197755903282399, abs=1e-12)
    assert rays[0.0]["residual"] == pytest.approx(-0.07174213357008696, abs=1e-12)
