# -*- coding: utf-8 -*-
"""SSAO 核采样与降噪

仅使用 Python 标准库，在深度缓冲上做遮蔽估计并抑制噪声。

约定：
- 深度值越大表示离相机越远；样本深度小于中心深度时视为潜在遮蔽体。
- 遮蔽值 AO ∈ [0, 1]，1 表示完全无遮蔽，0 表示完全遮蔽。
"""

import math
from dataclasses import dataclass

# 黄金比例（小数部分），用于低差异序列
_GOLDEN_FRAC = 0.6180339887498949
# 黄金角（弧度）
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))

# 深度突变判据：中心与样本的深度差超过场景深度范围的该比例时，
# 认为样本属于另一个物体（前景边缘），不计入遮蔽，避免黑边。
EDGE_DEPTH_RATIO = 0.1


@dataclass
class DepthBuffer:
    """深度缓冲。data 为 height 行 × width 列的浮点列表。"""
    width: int
    height: int
    data: list


def sample_kernel(count, radius, seed):
    """生成采样核。

    采样点按半径与旋转分布：半径用黄金比例低差异序列的平方根缩放，
    角度按黄金角旋转铺开，整体再按 seed 确定性旋转。

    核是"嵌套"的：count=N 的核是 count=N+1 的核的前缀，
    因此样本数变化时已有样本位置不动，结果不会跳变。
    同样的 (count, radius, seed) 永远得到同样的核。
    """
    if count < 1:
        raise ValueError("count 必须 >= 1")
    if radius <= 0:
        raise ValueError("radius 必须 > 0")
    # 由 seed 确定性地推出一个整体旋转角（不使用随机数发生器，保证可复现）
    rot = ((int(seed) * 2654435761) % 1000003) / 1000003.0 * 2.0 * math.pi
    kernel = []
    for i in range(count):
        t = ((i + 1) * _GOLDEN_FRAC) % 1.0          # (0,1) 内低差异分布
        r = radius * math.sqrt(t)                    # 面积均匀：半径按 sqrt 缩放
        a = i * _GOLDEN_ANGLE + rot                  # 黄金角旋转分布
        kernel.append((r * math.cos(a), r * math.sin(a)))
    return kernel


def compute_ao(buf, kernel):
    """算遮蔽。

    对每个像素，把核样本偏移到邻近像素（最近邻、边界钳制），
    样本深度 < 中心深度且深度差不超过边缘阈值时计入遮蔽，
    权重为 深度差 / 边缘阈值 的连续值（避免二值计数在样本数
    变化时产生量化跳变）。
    返回 height 行 × width 列的 AO 列表，值域 [0, 1]。

    边缘判据：深度差 > EDGE_DEPTH_RATIO × 场景深度范围 的样本被忽略，
    因此深度突变的边缘不会产生黑边。
    """
    w, h, data = buf.width, buf.height, buf.data
    if not kernel:
        raise ValueError("kernel 不能为空")
    flat = [v for row in data for v in row]
    depth_range = max(flat) - min(flat)
    edge_thresh = EDGE_DEPTH_RATIO * depth_range
    eps = 1e-12
    n = len(kernel)
    inv_thresh = 1.0 / edge_thresh if edge_thresh > 0.0 else 0.0
    ao = []
    for y in range(h):
        row = []
        for x in range(w):
            c = data[y][x]
            occ = 0.0
            for dx, dy in kernel:
                sx = min(max(int(round(x + dx)), 0), w - 1)
                sy = min(max(int(round(y + dy)), 0), h - 1)
                d = c - data[sy][sx]
                if d > eps and d <= edge_thresh:
                    occ += d * inv_thresh
            row.append(1.0 - occ / n)
        ao.append(row)
    return ao


def denoise(ao, sigma_s, sigma_r):
    """降噪：对 AO 图做双边滤波（保边缘），再做均值校正。

    - 空间权重为高斯（sigma_s），值域权重为高斯（sigma_r）：
      AO 差值大的像素（边缘两侧）权重小，因此边缘不被抹平。
    - 滤波后把整体均值平移回滤波前均值，保证平均遮蔽值不漂移。
    """
    if sigma_s <= 0 or sigma_r <= 0:
        raise ValueError("sigma_s / sigma_r 必须 > 0")
    h = len(ao)
    w = len(ao[0])
    rad = max(1, int(math.ceil(2.0 * sigma_s)))
    ss2 = 2.0 * sigma_s * sigma_s
    sr2 = 2.0 * sigma_r * sigma_r
    out = []
    for y in range(h):
        row = []
        y0, y1 = max(0, y - rad), min(h, y + rad + 1)
        for x in range(w):
            c = ao[y][x]
            wsum = 0.0
            vsum = 0.0
            x0, x1 = max(0, x - rad), min(w, x + rad + 1)
            for j in range(y0, y1):
                dy2 = (j - y) * (j - y)
                aor = ao[j]
                for i in range(x0, x1):
                    ds2 = (i - x) * (i - x) + dy2
                    dr = aor[i] - c
                    wt = math.exp(-ds2 / ss2) * math.exp(-dr * dr / sr2)
                    wsum += wt
                    vsum += wt * aor[i]
            row.append(vsum / wsum)
        out.append(row)
    # 均值校正：把滤波造成的整体偏移加回来
    count = w * h
    m1 = sum(v for row in ao for v in row) / count
    m2 = sum(v for row in out for v in row) / count
    delta = m1 - m2
    return [[v + delta for v in row] for row in out]


def load_depth(path):
    """读入深度文件：首行 "width height"，随后 height 行、每行 width 个浮点。"""
    with open(path, encoding="utf-8") as f:
        tokens = f.read().split()
    if len(tokens) < 2:
        raise ValueError("深度文件缺少宽高头")
    w, h = int(tokens[0]), int(tokens[1])
    vals = [float(t) for t in tokens[2:]]
    if len(vals) != w * h:
        raise ValueError("深度数据长度 %d 与 %dx%d 不符" % (len(vals), w, h))
    data = [vals[r * w:(r + 1) * w] for r in range(h)]
    return DepthBuffer(w, h, data)


def save_ao(path, ao):
    """把 AO 图按行写盘，每行空格分隔的定点小数。"""
    with open(path, "w", encoding="utf-8") as f:
        for row in ao:
            f.write(" ".join("%.6f" % v for v in row) + "\n")
