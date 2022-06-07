import cv2
import numpy as np
from cv2 import pyrDown


def select_reference_frame():
    """
    从多张RAW图中选则参考帧

    :return:
    """
    return


def align():
    """
    计算其他raw图的参考raw图的偏差
    
    :return: 
    """
    return


def gaussian_alignment():
    """
    四层从粗到细的高斯金字塔对齐
    :return:
    """





if __name__ == "__main__":
    img = np.random.randint(low=0, high=255, size=(3000, 3000), dtype="uint8")
