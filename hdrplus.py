'''
Author: kirby00001 t3273139612@hotmail.com
Date: 2022-05-21 15:19:54
LastEditors: kirby00001 t3273139612@hotmail.com
LastEditTime: 2022-06-11 14:09:34
FilePath: \HDRplus\hdrplus.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import rawpy
from alignment.alignment import burst_align

if __name__ == "__main__":
    ref_index = 0
    images = []
    for dir_path, dir_names, file_names in os.walk("./bursts/0006_20160722_115157_431/"):
        for file_name in file_names:
            if file_name.endswith(".dng"):
                file_path = os.path.join(dir_path, file_name)
                with rawpy.imread(file_path) as raw:
                    bayer_image = raw.raw_image.copy()
                    images.append(bayer_image)
    aligned_bursts_tiles, padding_size = burst_align(images, ref_index)
    print(aligned_bursts_tiles[-1, -1])
    print(padding_size)