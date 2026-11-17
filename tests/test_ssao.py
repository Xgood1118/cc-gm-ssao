# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ssao as mod  # noqa: E402
ssao = mod

def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_hex(path):
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def test_flat_gives_uniform_ao():
    buf = ssao.load_depth("data/depth.txt")
    ao = ssao.compute_ao(buf, ssao.sample_kernel(8, 4, 7))
    assert len(ao) == buf.height
    vals = [v for row in ao for v in row]
    assert max(vals) - min(vals) < 0.05


def test_denoise_keeps_mean():
    buf = ssao.load_depth("data/depth.txt")
    ao = ssao.compute_ao(buf, ssao.sample_kernel(8, 4, 7))
    out = ssao.denoise(ao, 2.0, 0.1)
    m1 = sum(v for row in ao for v in row) / (buf.width * buf.height)
    m2 = sum(v for row in out for v in row) / (buf.width * buf.height)
    assert abs(m1 - m2) < 0.02
