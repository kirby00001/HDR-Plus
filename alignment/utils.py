import cv2
from cv2 import pyrDown, matchTemplate, minMaxLoc

import numpy as np
from numpy.lib.stride_tricks import as_strided


def downSample(fullSize):
    """
    将拜耳RAW图进行平均，转化成灰度图

    :param fullSize: raw file in ndarray format
    :return: 平均后的灰度图
    """
    return fullSize[0::2, 0::2] + fullSize[1::2, 0::2] + fullSize[0::2, 1::2] + fullSize[1::2, 1::2] // 4


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


if __name__ == "__main__":
    img = np.random.randint(low=0, high=255, size=(759, 1012), dtype="uint8")
    print(get_tiles(img, tile_size=16, step=8).shape)
    # pyr = gaussian_pyramid(img)
    # for level in pyr:
    #     print(level.shape)
