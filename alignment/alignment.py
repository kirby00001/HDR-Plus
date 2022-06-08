import cv2
import numpy as np
from cv2 import pyrDown
from .utils import get_tiles, gaussian_pyramid


def select_reference_frame(burstPath, imageList, options):
    """
    从多张RAW图中选则参考帧

    :param burstPath:
    :return:
    """
    return


def gaussian_alignment(ref_image, alt_images, tile_size, search_radius_list=[4, 4, 2, 1], sampling_ratios=[1, 2, 4, 4],
                       norm_list=["L2", "L2", "L2", "L1"]):
    """
    四层从粗到细的高斯金字塔对齐

    :param ref_image: 参考图像
    :param alt_images: 候选图像
    :param tile_size: 图块大小
    :param search_radius_list: 金字塔每层的搜索半径
    :param norm_list: 金字塔每层的标准化方式
    :return:
    """
    up_sample_ratios = [1] + sampling_ratios[:0:-1]
    # 划分参考图像
    ref_tiles = get_tiles(ref_image, tile_size)
    aligned_tiles = np.empty(shape=((len(alt_images),) + ref_tiles.shape), dtype=ref_image.dtype)
    aligned_tiles[0] = ref_tiles
    motion_vectors = np.empty((len(alt_images), ref_tiles.shape[0], ref_tiles.shape[1], 2))
    # 构造四层高斯金字塔
    ref_pyr = gaussian_pyramid(ref_image, sampling_ratios)
    # 根据参考图像，对齐每一个候选图像
    for i, alt_image in enumerate(alt_images):
        alt_pyr = gaussian_pyramid(alt_image, sampling_ratios)
        motion_vector = None
        for ref_level, alt_level, search_radius, norm, up_sample_ratio in zip(ref_pyr, alt_pyr,
                                                                              search_radius_list,
                                                                              norm_list,
                                                                              up_sample_ratios):
            motion_vector = level_alignment(ref_level, alt_level, tile_size, search_radius, norm,
                                            pre_align=motion_vector,
                                            pre_tile_size=tile_size,
                                            up_sample_ratio=up_sample_ratio)


def level_alignment(ref_level, alt_level, tile_size, search_radius, norm, pre_align=None, pre_tile_size=None,
                    up_sample_ratio=None):
    """
    计算高斯金字塔的一层的运动向量

    :param ref_level: 参考图对应的高斯金字塔的某一层
    :param alt_level: 备用图对应的高斯金字塔的某一层
    :param tile_size: 图块的大小
    :param search_radius: 在备用图上的搜索半径
    :param norm: 模板匹配程度的标准化方法 L1, L2
    :param pre_align: 上一层对齐计算的运动向量
    :param pre_tile_size: 上一层的图块大小
    :param up_sample_ratio: 上采样倍率
    :return: 备用图相对参考图的运动向量
    """
    ref_tiles = get_tiles(ref_level, tile_size)
    # 如果存在上层对齐计算的运动向量，将其上采样到当前层级
    if pre_align is None:
        upsampled_pre_align = np.zeros(shape=(ref_tiles.shape[0], ref_tiles.shape[1], 2))
    else:
        upsampled_pre_align = up_sample_previous_alignment(ref_level, alt_level, pre_align, tile_size, pre_tile_size,
                                                           up_sample_ratio)
    # 计算搜索范围

    # 模板匹配
    # 获取最小距离对应的块的索引
    # 计算当前层每块的偏移量
    offsets = np.zeros((upsampled_pre_align.shape[0] * upsampled_pre_align.shape[1]), dtype=np.float32)

    return offsets


def up_sample_previous_alignment(ref_level, alt_level, pre_align, tile_size, pre_tile_size, up_sample_ratio):
    """
    将上个层级的对齐得到的运动向量上采样到当前层级

    :param ref_level: 当前层级的参考图
    :param alt_level: 当前层级的备选图
    :param pre_align: 上层级的运动向量
    :param tile_size: 当前层级的图块大小
    :param pre_tile_size: 上层级的图块大小
    :param up_sample_ratio: 上采样倍率
    :return: 上采样后的运动向量
    """
    # 扩大运动向量规模
    pre_align *= up_sample_ratio
    # 当前层图块的数量/上一层图块的数量
    repeat_ratio = up_sample_ratio // (tile_size // pre_tile_size)
    up_sampled_align = pre_align.repeat(repeat_ratio, axis=0).repeat(repeat_ratio, axis=1)
    h, w = up_sampled_align.shape[0], up_sampled_align.shape[1]
    # 选择三个最临近的块当作候选块
    tile = np.empty(shape=(repeat_ratio, repeat_ratio, 2, 2), dtype=np.int)
    index = repeat_ratio // 2
    tile[:index, :index] = [[-1, 0],
                            [0, -1]]
    tile[:index, index:] = [[-1, 0],
                            [0, 1]]
    tile[index:, :index] = [[1, 0],
                            [0, -1]]
    tile[index:, index:] = [[1, 0],
                            [0, 1]]

    neighborsMask = np.tile(tile, (
        up_sampled_align.shape[0] // repeat_ratio,
        up_sampled_align.shape[1] // repeat_ratio,
        1, 1))

    ti1 = np.repeat(
        np.clip(2 + np.arange(h) // repeat_ratio + neighborsMask[:, 0, 0, 0], 0, pre_align.shape[0] - 1).reshape(h, 1),
        w, axis=1).reshape(h * w)

    ti2 = np.repeat(
        np.clip(2 + np.arange(h) // repeat_ratio + neighborsMask[:, 0, 1, 0], 0, pre_align.shape[0] - 1).reshape(h, 1),
        w, axis=1).reshape(h * w)

    tj1 = np.repeat(
        np.clip(2 + np.arange(w) // repeat_ratio + neighborsMask[0, :, 0, 1], 0, pre_align.shape[1] - 1).reshape(1, w),
        h, axis=0).reshape(h * w)

    tj2 = np.repeat(
        np.clip(2 + np.arange(w) // repeat_ratio + neighborsMask[0, :, 1, 1], 0, pre_align.shape[1] - 1).reshape(1, w),
        h, axis=0).reshape(h * w)

    # Extract the previously estimated motion vectors associated with those neighbors
    ppa1 = pre_align[ti1, tj1].reshape((h, w, 2))
    ppa2 = pre_align[ti2, tj2].reshape((h, w, 2))

    # Get all possible tiles in the reference and alternate pyramid level
    refTiles = get_tiles(ref_level, tile_size, 1)
    altTiles = get_tiles(alt_level, tile_size, 1)

    # Compute the distance between the reference tile and the tiles that we're actually interested in
    d0 = computeTilesDistanceL1_(refTiles, altTiles, up_sampled_align).reshape(h * w)
    d1 = computeTilesDistanceL1_(refTiles, altTiles, ppa1).reshape(h * w)
    d2 = computeTilesDistanceL1_(refTiles, altTiles, ppa2).reshape(h * w)

    # Get the candidate motion vectors
    candidateAlignments = np.empty((h * w, 3, 2))
    candidateAlignments[:, 0, :] = up_sampled_align.reshape(h * w, 2)
    candidateAlignments[:, 1, :] = ppa1.reshape(h * w, 2)
    candidateAlignments[:, 2, :] = ppa2.reshape(h * w, 2)

    # Select the best candidates by keeping those that minimize the L1 distance with the reference tiles
    selectedAlignments = candidateAlignments[np.arange(h * w), np.argmin([d0, d1, d2], axis=0)].reshape((h, w, 2))
    # 在候选块中计算
    return up_sampled_align


if __name__ == "__main__":
    img = np.random.randint(low=0, high=255, size=(3000, 3000), dtype="uint8")
    print((8,) + img.shape)
