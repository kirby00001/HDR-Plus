import cv2
import numpy as np
from cv2 import pyrDown


def downSample(fullSize):
    """
    Down sample full size image into a grayscale image by averaging 2x2 block
    :param fullSize: raw file in ndarray format
    :return: grayscale image
    """
    return fullSize[0::2, 0::2] + fullSize[1::2, 0::2] + fullSize[0::2, 1::2] + fullSize[1::2, 1::2] // 4


def gaussian_pyramid(image, size_list=None):
    """
    根据输入图像和指定的缩小倍数创建高斯金字塔。

    :param image: 输入图像
    :param size_list: 每层相对上层缩小的倍数
    :return: ndarray list
    """
    if size_list is None:
        size_list = [4, 2, 4]
    _pyr = [image]
    cur = image
    for size in size_list:
        for _ in range(size // 2):
            cur = pyrDown(src=cur)
        _pyr.append(cur)
    return _pyr


def split(templet, size):
    return


def offset(tmp_block, reference, init_offset=None):
    return offset


def alignment(reference, templet, range_list=None):
    """
    计算两个图像间的偏移量

    :param reference: 参考图像
    :param templet: 需要对齐的图像
    :param range_list: 每层的最大偏移量
    :return: 需要对其图像中每个像素的偏移量(x, y) [n*n*2]
    """
    ref_pry = gaussian_pyramid(reference, [4, 2, 4])
    tmp_pry = gaussian_pyramid(templet, [4, 2, 4])
    offset = []
    for ref_layer, tmp_layer in zip(ref_pry, tmp_pry):
        # 划分tmp layer
        blocks = split(tmp_layer, (16, 16))
        # 对每个block计算偏移量
        offset = offset(blocks, ref_layer, offset)
    # 返回偏移量矩阵
    return offset
