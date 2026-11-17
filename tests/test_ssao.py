# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssao as mod  # noqa: E402
ssao = mod


def make_buf(rows):
    return ssao.DepthBuffer(len(rows[0]), len(rows), [list(r) for r in rows])


def slope_buf(w=16, h=16, step=0.01):
    """缓坡深度：相邻像素深度差小，能产生真实遮蔽。"""
    return make_buf([[x * step + y * step for x in range(w)] for y in range(h)])


def step_buf(w=16, h=16, lo=1.0, hi=2.0):
    """纯阶梯：左半 lo，右半 hi，深度突变。"""
    return make_buf([[lo if x < w // 2 else hi for x in range(w)] for y in range(h)])


def flat_buf(w=16, h=16, v=1.0):
    return make_buf([[v] * w for _ in range(h)])


def mean(ao):
    n = sum(len(row) for row in ao)
    return sum(v for row in ao for v in row) / n


# ---------- 遮蔽 ----------

def test_flat_gives_uniform_ao():
    buf = ssao.load_depth("data/depth.txt")
    ao = ssao.compute_ao(buf, ssao.sample_kernel(8, 4, 7))
    assert len(ao) == buf.height
    vals = [v for row in ao for v in row]
    assert max(vals) - min(vals) < 0.05


def test_flat_plane_is_unoccluded():
    ao = ssao.compute_ao(flat_buf(), ssao.sample_kernel(16, 4, 7))
    assert all(v == 1.0 for row in ao for v in row)


def test_zero_depth_is_unoccluded():
    """深度全为 0：确定结果——无遮蔽。"""
    ao = ssao.compute_ao(flat_buf(v=0.0), ssao.sample_kernel(16, 4, 7))
    assert all(v == 1.0 for row in ao for v in row)


def test_step_has_no_dark_edge():
    """纯阶梯：深度突变边缘不得出现黑边，边缘像素 AO 不低于内部。"""
    buf = step_buf()
    ao = ssao.compute_ao(buf, ssao.sample_kernel(16, 4, 7))
    vals = [v for row in ao for v in row]
    assert min(vals) >= 0.95
    # 边缘列（x = w/2-1 与 w/2）与内部一致
    w = buf.width
    for row in ao:
        assert row[w // 2 - 1] >= 0.95
        assert row[w // 2] >= 0.95


def test_sample_count_insensitive():
    """样本数变化时平均遮蔽不能跳变过大。"""
    buf = slope_buf()
    means = []
    for n in (8, 16, 32, 64):
        ao = ssao.compute_ao(buf, ssao.sample_kernel(n, 4, 7))
        means.append(mean(ao))
    for a, b in zip(means, means[1:]):
        assert abs(a - b) < 0.05


def test_kernel_is_nested():
    """核嵌套：count=N 是 count=N+1 的前缀，这是样本数不敏感的原因。"""
    k8 = ssao.sample_kernel(8, 4, 7)
    k9 = ssao.sample_kernel(9, 4, 7)
    assert k8 == k9[:8]


def test_deterministic():
    """同样输入两次结果一致。"""
    buf = slope_buf()
    k1 = ssao.sample_kernel(16, 4, 7)
    k2 = ssao.sample_kernel(16, 4, 7)
    assert k1 == k2
    ao1 = ssao.compute_ao(buf, k1)
    ao2 = ssao.compute_ao(buf, k2)
    assert ao1 == ao2
    assert ssao.denoise(ao1, 2.0, 0.1) == ssao.denoise(ao2, 2.0, 0.1)


# ---------- 降噪 ----------

def test_denoise_keeps_mean():
    buf = ssao.load_depth("data/depth.txt")
    ao = ssao.compute_ao(buf, ssao.sample_kernel(8, 4, 7))
    out = ssao.denoise(ao, 2.0, 0.1)
    m1 = sum(v for row in ao for v in row) / (buf.width * buf.height)
    m2 = sum(v for row in out for v in row) / (buf.width * buf.height)
    assert abs(m1 - m2) < 0.02


def test_denoise_keeps_mean_nonuniform():
    """非均匀 AO 下降噪前后均值不漂移。"""
    ao = ssao.compute_ao(slope_buf(), ssao.sample_kernel(16, 4, 7))
    out = ssao.denoise(ao, 2.0, 0.1)
    assert abs(mean(ao) - mean(out)) < 1e-9


def test_denoise_preserves_edges():
    """边缘保留：AO 阶梯经降噪后边缘对比度保留 >= 80%。"""
    w, h = 16, 16
    ao = [[0.4 if x < w // 2 else 0.9 for x in range(w)] for y in range(h)]
    out = ssao.denoise(ao, 2.0, 0.1)
    mid = w // 2
    before = ao[0][mid] - ao[0][mid - 1]
    # 取所有行中边缘两侧差的最小值，最差点也要保留 80%
    after = min(row[mid] - row[mid - 1] for row in out)
    assert after >= 0.8 * before


def test_denoise_smooths_noise_not_edges():
    """降噪要抑制平坦区噪声，同时不抹平边缘。"""
    w, h = 16, 16
    ao = []
    for y in range(h):
        row = []
        for x in range(w):
            base = 0.4 if x < w // 2 else 0.9
            # 确定性伪噪声
            noise = 0.05 if (x * 7 + y * 13) % 2 == 0 else -0.05
            row.append(base + noise)
        ao.append(row)
    out = ssao.denoise(ao, 2.0, 0.1)
    mid = w // 2
    # 平坦区（远离边缘）噪声幅度应下降
    def flat_spread(img, xs):
        vals = [img[y][x] for y in range(2, h - 2) for x in xs]
        return max(vals) - min(vals)
    assert flat_spread(out, range(2, mid - 2)) < flat_spread(ao, range(2, mid - 2))
    # 边缘对比度仍在
    assert out[h // 2][mid] - out[h // 2][mid - 1] > 0.4


# ---------- 边界 ----------

def test_step_deterministic_and_bounded():
    """纯阶梯输入有确定结果，且两次运行一致、值域合法。"""
    buf = step_buf()
    k = ssao.sample_kernel(16, 4, 7)
    ao1 = ssao.compute_ao(buf, k)
    ao2 = ssao.compute_ao(buf, k)
    assert ao1 == ao2
    assert all(0.0 <= v <= 1.0 for row in ao1 for v in row)
