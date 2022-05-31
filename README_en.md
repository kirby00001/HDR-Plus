# HDR plus python implementation
## 1. Pipline Overview
<img src="misc/Pipeline.png" style="zoom:70%;" />

## 2. Aligning Frames
- [ ] Reference frame selection
- [ ] Hierarchical alignment
### 2.1 Reference frame selection
根据绿色通道的梯度，在前3帧中选取最锐利的。
### 2.2 Handling raw images
输入是Bayer Raw图，四颜色平面是欠采样的，为了解决这个问题，我们假设偏移量为2像素的倍数。
四色平面：R、G、G、B
欠采样：每平面只有25%有值
通过平均2x2大小的RGGB下采样到3Mpix，再进行对齐
### 2.3 Hierarchical alignment

## 3. Merge
- [ ] Merge
## 4. Other Postprocessing