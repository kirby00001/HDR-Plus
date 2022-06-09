from asyncio.windows_events import NULL
from re import M
import cv2
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from cv2 import pyrDown, matchTemplate, minMaxLoc


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


def computeTilesDistanceL1_(ref_tiles, alt_tiles, offsets, result):
    """
    计算采用该偏移量后，参考图块和备选图块之间的L1距离，用作评估偏移量的指标

    :param ref_tiles: 参考图块
    :param alt_tiles: 备选图块
    :param offsets: 偏移量
    :param result: 结果
    :return: 
    """
    # Dimension
    m, n, tile_size, _ = ref_tiles.shape
    h, w, _ = offsets.shape
    # 遍历所有对齐的图块
    for i in range(h):
        for j in range(w):
            # 偏移量
            i_offset = offsets[i, j, 0]
            j_offset = offsets[i, j, 1]
            # 参考图块的索引
            ref_i = i * (tile_size // 2)
            ref_j = j * (tile_size // 2)
            # 对应图块的索引
            di = ref_i + int(i_offset + (0.5 if i_offset >= 0 else -0.5))
            dj = ref_j + int(j_offset + (0.5 if j_offset >= 0 else -0.5))
            # Clip the position
            di = 0 if di < 0 else (m - 1 if di > m - 1 else di)
            dj = 0 if dj < 0 else (n - 1 if dj > n - 1 else dj)
            # 计算L1距离
            dst = 0
            for p in range(tile_size):
                for q in range(tile_size):
                    dst += math.fabs(ref_tiles[ref_i, ref_j, p, q] - alt_tiles[di, dj, p, q])
            # Store the result
            result[i, j] = dst


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


if __name__ == "__main__":
    img = np.random.randint(low=0, high=255, size=(759, 1012), dtype="uint8")
    print(get_tiles(img, tile_size=16, step=8).shape)
    # pyr = gaussian_pyramid(img)
    # for level in pyr:
    #     print(level.shape)
