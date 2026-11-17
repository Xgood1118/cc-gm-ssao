# -*- coding: utf-8 -*-
"""SSAO 核采样与降噪"""

from dataclasses import dataclass


@dataclass
class DepthBuffer:
    """深度缓冲。"""
    width: int
    height: int
    data: list


def sample_kernel(count, radius, seed):
    """生成采样核"""
    raise NotImplementedError("sample_kernel 尚未实现")


def compute_ao(buf, kernel):
    """算遮蔽"""
    raise NotImplementedError("compute_ao 尚未实现")


def denoise(ao, sigma_s, sigma_r):
    """降噪"""
    raise NotImplementedError("denoise 尚未实现")

