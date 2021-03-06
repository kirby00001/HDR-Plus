import numpy as np
from .utils import down_sample, get_tiles, gaussian_pyramid, match_templates, select_offsets, compute_l1_distance_with_pre_alinment, get_aligned_tiles, compute_subpixel_offset


def select_reference_frame(burstPath, imageList, options):
    """
    从多张RAW图中选则参考帧

    :param burstPath:
    :return:
    """
    return


def burst_align(images, ref_index, tile_sizes=[8, 16, 16, 16]):
    """
    对齐burst RAW图
    
    :param images: 所有图像
    :param ref_index: 参考图像的索引
    :param tile_sizes: 每层的图块大小
    :return: 对齐后的图块，填充大小
    """
    h, w = images[ref_index].shape
    # 在RAW图层级上填充图像，让图块能包括所有像素
    bayer_tile_size = tile_sizes[-1] * 2
    padding_patches_h = (bayer_tile_size - h % (bayer_tile_size)) * (h % (bayer_tile_size) != 0)
    padding_patched_w = (bayer_tile_size - w % (bayer_tile_size)) * (w % (bayer_tile_size) != 0)
    # 零填充，防止图像边缘产生伪影
    padding_overlap_h = padding_overlap_w = bayer_tile_size // 2
    # 上
    padding_top = padding_overlap_h
    # 下
    padding_bottom = padding_overlap_h + padding_patches_h
    # 左
    padding_left = padding_overlap_w
    # 右
    padding_right = padding_overlap_w + padding_patched_w
    # 填充图像
    padded_images = [
        np.pad(im, ((padding_top, padding_bottom), (padding_left, padding_right)), 'symmetric')
        for im in images
    ]
    # 选择参考图像和备选图像
    ref_image = padded_images[ref_index]
    alt_images = [image for i, image in enumerate(padded_images) if i != ref_index]
    # 进行高斯金字塔对齐
    motion_vectors, aligned_tiles = gaussian_align(ref_image, alt_images, tile_sizes)
    padding = (padding_top, padding_bottom, padding_left, padding_right)
    return aligned_tiles, padding


def gaussian_align(ref_image,
                   alt_images,
                   tile_sizes,
                   subpixel_list=[True, True, True, False],
                   search_radius_list=[4, 4, 2, 1],
                   sampling_ratios=[1, 2, 4, 4],
                   norm_list=["L2", "L2", "L2", "L1"]):
    """
    四层从粗到细的高斯金字塔对齐

    :param ref_image: 参考图像
    :param alt_images: 候选图像
    :param tile_sizes: 每层图块大小
    :param search_radius_list: 金字塔每层的搜索半径
    :param norm_list: 金字塔每层的标准化方式
    :param sampling_ratios: 每层的采样倍率
    :return: 所有候选图的运动向量和所有对齐后的图块
    """
    up_sample_ratios = [1] + sampling_ratios[:0:-1]
    # RAW图转化成灰度图
    gray_ref_image = down_sample(ref_image)
    bayer_tile_size = tile_sizes[-1] * 2
    # 划分参考图像
    ref_tiles = get_tiles(ref_image, bayer_tile_size, bayer_tile_size // 2)
    aligned_tiles = np.empty(shape=((len(alt_images) + 1, ) + ref_tiles.shape),
                             dtype=ref_image.dtype)
    aligned_tiles[0] = ref_tiles
    motion_vectors = np.empty((len(alt_images), ref_tiles.shape[0], ref_tiles.shape[1], 2))
    # 以灰度参考图为基础，构造四层高斯金字塔
    ref_pyr = gaussian_pyramid(gray_ref_image, sampling_ratios)
    # 根据参考图像，对齐每一个候选图像
    pre_tile_sizes = [None] + tile_sizes[0:-1]
    for i, alt_image in enumerate(alt_images):
        # RAW图转为灰度图
        gray_alt_image = down_sample(alt_image)
        alt_pyr = gaussian_pyramid(gray_alt_image, sampling_ratios)
        motion_vector = None
        for ref_level, alt_level, tile_size, search_radius, norm, subpixel, pre_tile_size, up_sample_ratio in zip(
                ref_pyr,
                alt_pyr,
                tile_sizes,
                search_radius_list,
                norm_list,
                subpixel_list,
                pre_tile_sizes,
                up_sample_ratios,
        ):
            motion_vector = level_align(ref_level,
                                        alt_level,
                                        tile_size,
                                        search_radius,
                                        norm,
                                        subpixel,
                                        pre_align=motion_vector,
                                        pre_tile_size=pre_tile_size,
                                        up_sample_ratio=up_sample_ratio)
        # 从灰度图上采样到RAW图
        motion_vector = up_sample_previous_alignment(None, None, motion_vector, bayer_tile_size,
                                                     tile_size, 2)
        # 根据运动向量，得到对齐后的备选图块
        aligned_alt_tiles = get_aligned_tiles(alt_image, bayer_tile_size, motion_vector)
        # 保存运动向量和对齐后的图块
        motion_vectors[i] = motion_vector
        aligned_tiles[i + 1] = aligned_alt_tiles
    return motion_vectors, aligned_tiles


def level_align(ref_level,
                alt_level,
                tile_size,
                search_radius,
                norm,
                subpixel,
                pre_align=None,
                pre_tile_size=None,
                up_sample_ratio=None):
    """
    计算高斯金字塔的一层的运动向量

    :param ref_level: 参考图对应的高斯金字塔的某一层
    :param alt_level: 备用图对应的高斯金字塔的某一层
    :param tile_size: 图块的大小
    :param search_radius: 在备用图上的搜索半径
    :param norm: 模板匹配程度的标准化方法 L1, L2
    :param subpixel: 是否采用亚像素对齐 
    :param pre_align: 上一层对齐计算的运动向量
    :param pre_tile_size: 上一层的图块大小
    :param up_sample_ratio: 上采样倍率
    :return: 备用图相对参考图的运动向量
    """
    # 转换数据类型，为之后模板匹配计算准备
    ref_level = ref_level.astype(np.float32)
    alt_level = alt_level.astype(np.float32)
    # 提取瓦片
    ref_tiles = get_tiles(ref_level, tile_size, step=tile_size // 2)
    h, w = ref_tiles.shape[0], ref_tiles.shape[1]
    # 如果存在上层对齐计算的运动向量，将其上采样到当前层级
    if pre_align is None:
        upsampled_pre_align = np.zeros(shape=(h, w, 2))
    else:
        upsampled_pre_align = up_sample_previous_alignment(
            ref_level,
            alt_level,
            pre_align,
            tile_size,
            pre_tile_size,
            up_sample_ratio,
        )
    # 提取u0, v0
    u0 = upsampled_pre_align[:, :, 0].round().astype(np.int32)
    v0 = upsampled_pre_align[:, :, 1].round().astype(np.int32)
    # 处理边界问题，并计算搜索范围
    padded_alt_level = np.pad(alt_level, search_radius, mode="constant", constant_values=2**16 - 1)
    search_areas = get_tiles(padded_alt_level, tile_size + 2 * search_radius, step=1)
    # 根据上一层的初始化偏移量，计算搜索范围索引，修正搜索范围
    i = (np.arange(h).reshape(h, 1).repeat(w, axis=1).reshape(h, w) * tile_size // 2 + u0).reshape(
        h * w).clip(0, search_areas.shape[0] - 1)
    j = (np.arange(w).reshape(1, w).repeat(h, axis=0).reshape(h, w) * tile_size // 2 + v0).reshape(
        h * w).clip(0, search_areas.shape[1] - 1)
    search_areas = search_areas[i, j].reshape(h, w, search_areas.shape[2], search_areas.shape[3])
    # 模板匹配,计算距离
    distances = match_templates(ref_tiles, search_areas, norm)
    # 获取最小距离对应的块的位置，即相对运动向量
    offsets = select_offsets(distances)
    # 亚像素对齐
    if subpixel:
        offsets = offsets.reshape(h * w, 2)
        distances = distances.reshape(h * w, distances.shape[-1]**2)
        min_index = np.argmin(distances, axis=1).astype(np.uint16)
        subpixel_offsets = np.zeros_like(offsets)
        valid_min_index = np.logical_and(min_index < distances.shape[1] - 4, min_index >= 4)
        subpixel_offsets[valid_min_index] = compute_subpixel_offset(distances[valid_min_index],
                                                                    min_index[valid_min_index])
        offsets += subpixel_offsets
        offsets = offsets.reshape(h, w, 2)
    # 上面的相对运动向量是相对(u0,v0)-radius的，修正偏差
    offsets[:, :, 0] += (u0 - search_radius)
    offsets[:, :, 1] += (v0 - search_radius)
    return offsets


def up_sample_previous_alignment(ref_level, alt_level, pre_align, tile_size, pre_tile_size,
                                 up_sample_ratio):
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
    # 运动向量矩阵的扩大倍数 = 当前层图块的数量/上一层图块的数量
    repeat_ratio = up_sample_ratio // (tile_size // pre_tile_size)
    # 扩大运动向量矩阵
    up_sampled_alignment = pre_align.repeat(repeat_ratio, axis=0).repeat(repeat_ratio, axis=1)
    # 如果是灰度图到RAW图的上采样，直接返回结果
    if ref_level is None:
        return up_sampled_alignment
    # 上采样后，运动向量矩阵的高和宽
    h, w = up_sampled_alignment.shape[0], up_sampled_alignment.shape[1]
    # 在运动向量边界填充，解决分块计算邻居时的边界问题
    padded_pre_alignment = np.pad(pre_align, pad_width=((1, 1), (1, 1), (0, 0)), mode='edge')
    # 选择三个最临近的块当作候选块
    tile = np.empty(shape=(repeat_ratio, repeat_ratio, 2, 2), dtype=np.int)
    # 等分为四部分
    index = repeat_ratio // 2
    # 上、左
    tile[:index, :index] = [[-1, 0], [0, -1]]
    # 上、右
    tile[:index, index:] = [[-1, 0], [0, 1]]
    # 下、左
    tile[index:, :index] = [[1, 0], [0, -1]]
    # 下、右
    tile[index:, index:] = [[1, 0], [0, 1]]
    # 从一个图块扩大到整张图的大小
    neighbors_mask = np.tile(tile, (up_sampled_alignment.shape[0] // repeat_ratio,
                                    up_sampled_alignment.shape[1] // repeat_ratio, 1, 1))
    # 上下邻居的索引
    vertical_i = (1 + (np.arange(h) // repeat_ratio) + neighbors_mask[:, 0, 0, 0]).clip(
        0, padded_pre_alignment.shape[0] - 1).reshape(h, 1).repeat(w, axis=1).reshape(h * w)
    vertical_j = (1 + (np.arange(w) // repeat_ratio) + neighbors_mask[0, :, 0, 1]).clip(
        0, padded_pre_alignment.shape[1] - 1).reshape(1, w).repeat(h, axis=0).reshape(h * w)
    # 左右邻居的索引
    horizontal_i = (1 + (np.arange(h) // repeat_ratio) + neighbors_mask[:, 0, 1, 0]).clip(
        0, padded_pre_alignment.shape[0] - 1).reshape(h, 1).repeat(w, axis=1).reshape(h * w)
    horizontal_j = (1 + (np.arange(w) // repeat_ratio) + neighbors_mask[0, :, 1, 1]).clip(
        0, padded_pre_alignment.shape[1] - 1).reshape(1, w).repeat(h, axis=0).reshape(h * w)
    # 提取出对应的邻居的运动向量
    vertical_neighbours = padded_pre_alignment[vertical_i, vertical_j].reshape((h, w, 2))
    horizontal_neighbours = padded_pre_alignment[horizontal_i, horizontal_j].reshape((h, w, 2))
    # 获取所有可能的参考和候选图块
    ref_tiles = get_tiles(ref_level, tile_size, 1)
    alt_tiles = get_tiles(alt_level, tile_size, 1)
    # 根据偏移量，计算参考图块和候选图块的L1距离
    d0 = compute_l1_distance_with_pre_alinment(ref_tiles, alt_tiles,
                                               up_sampled_alignment).reshape(h * w)
    d1 = compute_l1_distance_with_pre_alinment(ref_tiles, alt_tiles,
                                               vertical_neighbours).reshape(h * w)
    d2 = compute_l1_distance_with_pre_alinment(ref_tiles, alt_tiles,
                                               horizontal_neighbours).reshape(h * w)
    # 构建候选运动向量
    candidate_alignment = np.empty((h * w, 3, 2))
    candidate_alignment[:, 0, :] = up_sampled_alignment.reshape(h * w, 2)
    candidate_alignment[:, 1, :] = vertical_neighbours.reshape(h * w, 2)
    candidate_alignment[:, 2, :] = horizontal_neighbours.reshape(h * w, 2)
    # 从三个候选运动向量中，选出能最小化L1距离的运动向量
    selected_alignment = candidate_alignment[np.arange(h * w),
                                             np.argmin([d0, d1, d2], axis=0)].reshape((h, w, 2))
    # 因为划分图块时，边缘图块被忽略，没有下层的对齐，需要被初始化为0
    final_h = ref_level.shape[0] // (tile_size // 2) - 1
    final_w = ref_level.shape[1] // (tile_size // 2) - 1
    if h < final_h or w < final_w:
        final_alignmnet = np.zeros(shape=(final_h, final_w, 2), dtype=np.int32)
        final_alignmnet[:h, :w] = selected_alignment
    else:
        final_alignmnet = selected_alignment
    return final_alignmnet


if __name__ == "__main__":
    img = np.random.randint(low=0, high=255, size=(3000, 3000), dtype="uint8")
    print((8, ) + img.shape)
