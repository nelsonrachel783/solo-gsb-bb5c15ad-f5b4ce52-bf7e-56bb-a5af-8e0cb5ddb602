"""波前光程直线拟合的大坐标尺度边界回归测试。

覆盖 app.geometry.linear_least_squares 与 POST /api/wavefront-evaluations：
- 无界面、观察线 x=1e200、中心方向 (1,0)、半角 30°、采样 3 条且 n_start=1：
  三条光线的观察线交点与光程均有限，但切向偏移约 ±5.8e199、光程约 1e200，
  旧实现直接累加 (x − mean_x)² 抛 OverflowError（接口 500）；尺度归一化后
  斜率、截距、逐光线残差与残差 RMS 均返回有限值；
- ±1e308 跨度（连坐标差值 2e308 本身都溢出为无穷）的点集同样给出有限拟合；
- 有效光线少于 3 条或偏移全相同仍返回 null；普通尺度结果与原直接离差平方
  公式保持一致（仅末位浮点舍入差异），既有语义不回退。
"""
from __future__ import annotations

import math

import pytest

from app.geometry import WAVEFRONT_MIN_FIT_RAYS, linear_least_squares

WAVEFRONT_PATH = "/api/wavefront-evaluations"


def _legacy_linear_least_squares(points):
    """修复前的直接离差平方实现，仅用于普通尺度下的数值对照。"""
    count = len(points)
    if count < WAVEFRONT_MIN_FIT_RAYS:
        return None
    x0 = points[0][0]
    if all(p[0] == x0 for p in points):
        return None
    mean_x = sum(p[0] for p in points) / count
    mean_y = sum(p[1] for p in points) / count
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    if sxx == 0.0:
        return None
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    residuals = [p[1] - (intercept + slope * p[0]) for p in points]
    rms = math.sqrt(sum(r * r for r in residuals) / count)
    return intercept, slope, residuals, rms


def test_huge_observation_line_wavefront_endpoint(client):
    """观察线 x=1e200、无界面、3 条光线：旧实现 500，现 200 且拟合全有限。"""
    body = {
        "start": {"x": 0.0, "y": 0.0},
        "center_direction": {"x": 1.0, "y": 0.0},
        "half_angle_deg": 30.0,
        "sample_count": 3,
        "n_start": 1.0,
        "interfaces": [],
        "observation_line": {
            "point": {"x": 1e200, "y": 0.0},
            "normal": {"x": 1.0, "y": 0.0},
        },
    }
    resp = client.post(WAVEFRONT_PATH, json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["ray_count"] == 3
    assert data["hit_count"] == 3
    assert math.isfinite(data["intercept"])
    assert data["slope"] == 0.0  # 三点关于 offset=0 对称
    assert math.isfinite(data["residual_rms"]) and data["residual_rms"] > 0.0
    # 截距即三光线光程均值：(2·X/cos30° + X)/3
    x_obs = 1e200
    expected = x_obs * (2.0 / math.cos(math.radians(30.0)) + 1.0) / 3.0
    assert data["intercept"] == pytest.approx(expected, rel=2e-16)
    # 边缘两光线残差相等为正、中心光线残差为负（光程曲线在拟合直线下方）
    residuals = [r["residual"] for r in data["rays"]]
    assert all(math.isfinite(r) for r in residuals)
    assert residuals[0] == pytest.approx(residuals[2], rel=2e-16)
    assert residuals[0] > 0.0 and residuals[1] < 0.0
    # 逐光线的交点、光程与偏移均有限
    for ray in data["rays"]:
        assert ray["status"] == "completed"
        assert math.isfinite(ray["optical_path"])
        assert math.isfinite(ray["offset"])


def test_geometry_max_scale_points_finite_fit():
    """±1e308 跨度（坐标差值 2e308 溢出为 inf）仍返回有限的拟合结果。"""
    s = 1e308
    y_edge = s * 1.1
    points = [(-s, y_edge), (0.0, s), (s, y_edge)]
    fit = linear_least_squares(points)
    assert fit is not None
    intercept, slope, residuals, rms = fit
    assert slope == 0.0
    assert math.isfinite(intercept)
    assert all(math.isfinite(r) for r in residuals)
    assert math.isfinite(rms)
    # 截距解析均值（系数先聚合，不先乘出溢出的 2.2e308）
    expected = (2.0 * (y_edge / s) + 1.0) / 3.0 * s
    assert intercept == pytest.approx(expected, rel=1e-15)
    assert residuals[1] < 0.0 < residuals[0]

    # ±1e308 精确直线：非零斜率可定、残差与 RMS 为 0
    line = [(-s, -1.1 * s), (0.0, 0.0), (s, 1.1 * s)]
    fit_line = linear_least_squares(line)
    assert fit_line is not None
    assert fit_line[1] == pytest.approx(1.1, abs=2e-16)
    assert max(abs(r) for r in fit_line[2]) == 0.0
    assert fit_line[3] == 0.0


def test_geometry_fewer_than_three_points_null():
    """有效光线不足 3 条：返回 None。"""
    assert linear_least_squares([]) is None
    assert linear_least_squares([(1.0, 1.0)]) is None
    assert linear_least_squares([(1.0, 1.0), (2.0, 2.0)]) is None


def test_geometry_identical_offsets_null():
    """偏移全相同（斜率不可定）：即便大尺度也返回 None。"""
    assert linear_least_squares([(3.0, 1.0), (3.0, 2.0), (3.0, 9.0)]) is None
    s = 1e300
    assert linear_least_squares([(s, 1.0), (s, s), (s, 2.0 * s)]) is None


def test_geometry_constant_optical_path_horizontal_fit():
    """光程全相同：精确水平直线，斜率 0、残差与 RMS 为 0。"""
    assert linear_least_squares([(1.0, 7.0), (2.0, 7.0), (9.0, 7.0)]) == (
        7.0,
        0.0,
        [0.0, 0.0, 0.0],
        0.0,
    )
    assert linear_least_squares([(1.0, 0.0), (2.0, 0.0), (9.0, 0.0)]) == (
        0.0,
        0.0,
        [0.0, 0.0, 0.0],
        0.0,
    )


def test_geometry_normal_scale_matches_legacy_formula():
    """普通尺度：归一化拟合与原离差平方公式逐量一致（末位舍入容差）。"""
    cases = [
        [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0)],
        [(-2.0, 4.0), (-1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (2.0, 4.0)],
        [(1.0, 10.0), (2.0, 9.0), (3.0, 11.0), (4.0, 10.5)],
        # 微小尺度同样不被归一化破坏
        [(1e-10, 2e-10), (2e-10, 4.1e-10), (3e-10, 5.9e-10)],
    ]
    for points in cases:
        legacy = _legacy_linear_least_squares(points)
        fit = linear_least_squares(points)
        assert legacy is not None and fit is not None
        a0, b0, r0, rms0 = legacy
        a1, b1, r1, rms1 = fit
        assert a1 == pytest.approx(a0, rel=1e-12, abs=1e-12)
        assert b1 == pytest.approx(b0, rel=1e-12, abs=1e-12)
        assert rms1 == pytest.approx(rms0, rel=1e-12, abs=1e-12)
        for old_r, new_r in zip(r0, r1):
            assert new_r == pytest.approx(old_r, rel=1e-12, abs=1e-12)


def test_wavefront_endpoint_less_than_three_valid_null(client):
    """采样 2 条（有效光线不足 3）：拟合字段为 null，光程/偏移仍返回。"""
    body = {
        "start": {"x": 0.0, "y": 0.0},
        "center_direction": {"x": 1.0, "y": 0.0},
        "half_angle_deg": 5.0,
        "sample_count": 2,
        "n_start": 1.0,
        "interfaces": [],
        "observation_line": {
            "point": {"x": 10.0, "y": 0.0},
            "normal": {"x": 1.0, "y": 0.0},
        },
    }
    resp = client.post(WAVEFRONT_PATH, json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["hit_count"] == 2
    assert data["intercept"] is None
    assert data["slope"] is None
    assert data["residual_rms"] is None
    assert all(r["residual"] is None for r in data["rays"])
    assert all(r["optical_path"] is not None and r["offset"] is not None for r in data["rays"])


def test_wavefront_endpoint_glass_slab_still_finite(client):
    """普通玻璃板既有场景：拟合正常给出有限截距/斜率/RMS（README 示例语义）。"""
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
    # 与 README 示例数值仅差末位 1～2 ULP（归一化公式的等价浮点舍入）
    assert data["intercept"] == pytest.approx(11.071742133570087, abs=5e-14)
    assert data["residual_rms"] == pytest.approx(0.06019273926764479, abs=5e-14)
    assert all(math.isfinite(r["residual"]) for r in data["rays"])
