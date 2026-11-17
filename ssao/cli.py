# -*- coding: utf-8 -*-
"""命令行入口：python -m ssao.cli data/depth.txt --radius 4 --samples 8 --out ao.txt"""

import argparse

import ssao as m


def _mean(ao):
    n = sum(len(row) for row in ao)
    return sum(v for row in ao for v in row) / n


def _edge_mask(buf):
    """深度突变像素：与右/下邻居的深度差超过边缘阈值。"""
    w, h, data = buf.width, buf.height, buf.data
    flat = [v for row in data for v in row]
    thresh = m.EDGE_DEPTH_RATIO * (max(flat) - min(flat))
    mask = [[False] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            c = data[y][x]
            if x + 1 < w and abs(c - data[y][x + 1]) > thresh:
                mask[y][x] = mask[y][x + 1] = True
            if y + 1 < h and abs(c - data[y + 1][x]) > thresh:
                mask[y][x] = mask[y + 1][x] = True
    return mask


def _edge_retention(before, after):
    """边缘保留度：降噪后/前的全图梯度幅度和之比（1 表示完全保留）。"""
    def grad_sum(ao):
        h = len(ao)
        w = len(ao[0])
        s = 0.0
        for y in range(h):
            for x in range(w):
                if x + 1 < w:
                    s += abs(ao[y][x + 1] - ao[y][x])
                if y + 1 < h:
                    s += abs(ao[y + 1][x] - ao[y][x])
        return s
    g0 = grad_sum(before)
    if g0 <= 0.0:
        return 1.0
    return grad_sum(after) / g0


def main():
    ap = argparse.ArgumentParser(description="SSAO 核采样与降噪")
    ap.add_argument("depth", help="深度文件路径")
    ap.add_argument("--radius", type=float, default=4.0)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sigma-s", type=float, default=2.0)
    ap.add_argument("--sigma-r", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    buf = m.load_depth(a.depth)
    kernel = m.sample_kernel(a.samples, a.radius, a.seed)
    ao = m.compute_ao(buf, kernel)
    out = m.denoise(ao, a.sigma_s, a.sigma_r)
    m.save_ao(a.out, out)

    mean_before = _mean(ao)
    mean_after = _mean(out)

    mask = _edge_mask(buf)
    edge_vals = [out[y][x] for y in range(buf.height) for x in range(buf.width) if mask[y][x]]
    edge_ao = sum(edge_vals) / len(edge_vals) if edge_vals else None

    retention = _edge_retention(ao, out)

    print("平均遮蔽: %.6f" % mean_after)
    if edge_ao is None:
        print("边缘处遮蔽: 无深度边缘（场景为全平面）")
    else:
        print("边缘处遮蔽: %.6f" % edge_ao)
    print("降噪前后均值差: %.6f" % (mean_after - mean_before))
    print("边缘保留度: %.4f" % retention)
    print("已写出: %s" % a.out)


if __name__ == "__main__":
    main()
