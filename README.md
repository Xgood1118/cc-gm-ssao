# SSAO 核采样与降噪

一个能在深度缓冲上做遮蔽估计并抑制噪声的 SSAO 实现

## 保留签名

- `DepthBuffer`
- `sample_kernel(count, radius, seed)`
- `compute_ao(buf, kernel)`
- `denoise(ao, sigma_s, sigma_r)`

## 自测

```
python -m pytest tests -q
python -m ssao.cli data/depth.txt --radius 4 --samples 8 --out ao.txt
```
