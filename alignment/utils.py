from turtle import st
import cv2
from cv2 import pyrDown, matchTemplate, minMaxLoc
from matplotlib.pyplot import axis

import numpy as np
from numpy.lib.stride_tricks import as_strided


def downSample(fullSize):
    """
    将拜耳RAW图进行平均，转化成灰度图

    :param fullSize: raw file in ndarray format
    :return: 平均后的灰度图
    """
    return fullSize[0::2, 0::2] + fullSize[1::2, 0::2] + fullSize[0::2,
                                                                  1::2] + fullSize[1::2, 1::2] // 4


def get_tiles(image, tile_size, step):
    """
    将图像划分为图块, 图块间存在重叠部分

    :param image: 被划分的图像
    :param tile_size: 图块大小
    :param step: 划分块时的位移大小 s
    :return: 被划分的图块
    """
    image_shape = image.shape
    tile_shape = (tile_size, tile_size)
    shape = tuple((np.array(image_shape) - tile_size) // step + 1) + tile_shape

    image_strides = image.strides
    strides = tuple(np.array(image_strides) * step) + image_strides
    tiles = as_strided(image, shape=shape, strides=strides)

    return tiles


def gaussian_pyramid(image, sampling_ratios=[1, 2, 4, 4]):
    """
    创建四层从粗到细的高斯金字塔。

    :param image: 输入图像
    :param sampling_ratios: 每层相对上层缩小的倍数
    :return: ndarray list
    """
    current = image
    pyramid = []
    for size in sampling_ratios:
        if size == 1:
            pyramid.append(current)
        else:
            for _ in range(size // 2):
                current = pyrDown(src=current, borderType=cv2.BORDER_REFLECT)
            pyramid.append(current)
    return pyramid[::-1]


def compute_l1_distance_with_pre_alinment(ref_tiles, alt_tiles, offsets, result):
    """
    计算采用该偏移量后，参考图块和备选图块之间的L1距离，用作评估偏移量的指标

    :param ref_tiles: 参考图块
    :param alt_tiles: 备选图块
    :param offsets: 偏移量
    :param result: 结果
    :return: 
    """
    # Dimension
    h, w, _ = offsets.shape
    m, n, tile_size, _ = ref_tiles.shape
    result = np.empty(shape=(h, w), dtype=np.float32)
    # 遍历所有对齐的图块
    for i in range(h):
        for j in range(w):
            # 偏移量
            offset_i = offsets[i, j, 0]
            offset_j = offsets[i, j, 1]
            # 参考图块的索引,步长为tile_size // 2
            ref_i = i * (tile_size // 2)
            ref_j = j * (tile_size // 2)
            # 对应图块的索引
            alt_i = ref_i + offset_i
            alt_j = ref_j + offset_j
            # 限制索引范围
            if alt_i < 0:
                alt_i = 0
            elif alt_i > m - 1:
                alt_i = m - 1
            if alt_j < 0:
                alt_j = 0
            elif alt_j > n - 1:
                alt_j = n - 1
            # 计算L1距离
            l1_distance = np.sum(
                np.absolute(ref_tiles[ref_i, ref_j, :, :] - alt_tiles[alt_i, alt_j, :, :]))
            # Store the result
            result[i, j] = l1_distance
    return result


def match_templates(ref_tiles, search_areas, mode):
    if mode == "L2":
        mode = cv2.TM_SQDIFF
    h, w, _, _ = ref_tiles.shape
    distances = np.empty(shape=(h, w, -1, -1), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            distances[h, w, :, :] = matchTemplate(image=search_areas[i, j, :, :],
                                                  templ=ref_tiles[i, j, :, :],
                                                  method=mode)
    return distances


def select_offsets(distance):

    h, w, _, _ = distance.shape
    offsets = np.empty(shape=[h, w, 2], dtype=np.int32)
    for i in range(h):
        for j in range(w):
            _, _, min_index, _ = minMaxLoc(distance[h, w, :, :])
            offsets[h, w, :] = np.array(min_index)
    return offsets


def get_aligned_tiles(image, tile_size, motion_vectors):
    # 重叠一半瓦片划分出的瓦片矩阵的高度和宽度
    h, w = image.shape[0] // (tile_size // 2) - 1, image.shape[1] // (tile_size // 2) - 1
    # 有运动向量的瓦片数量: hm*wm (<= h*w)
    hm, wm, _ = motion_vectors.shape
    # 从图像中提取出所有可能的瓦片
    image_tiles = get_tiles(image, tile_size, step=1)
    # 获取施加运动向量后的瓦片索引
    motion_i = ((np.arange(hm) * tile_size // 2).reshape(hm, 1).repeat(wm, axis=1) +
                motion_vectors[:, :, 0]).clip(0, image.shape[0] - tile_size)
    motion_j = ((np.arange(wm) * tile_size // 2).reshape(1, wm).repeat(hm, axis=0) +
                motion_vectors[:, :, 1]).clip(0, image.shape[1] - tile_size)
    # 没有施加运动向量的瓦片索引
    i = (np.arange(h) * tile_size // 2).reshape(h, 1).repeat(w, axis=1)
    j = (np.arange(w) * tile_size // 2).reshape(1, w).repeat(h, axis=0)
    # 没有对应运动向量的会保持原位置
    i[:motion_i.shape[0], :motion_i.shape[1]] = motion_i
    j[:motion_j.shape[0], :motion_j.shape[1]] = motion_j
    # 保存有效的瓦片 (step = tileSize // 2)
    alignedTiles = image_tiles[i.reshape(h * w), j.reshape(h * w)].reshape(
        (h, w, tile_size, tile_size))
    return alignedTiles


if __name__ == "__main__":
    img = np.random.randint(low=0, high=255, size=(759, 1012), dtype="uint8")
    print(get_tiles(img, tile_size=16, step=8).shape)
    pyr = gaussian_pyramid(img)
    for level in pyr:
        print(level.shape)
