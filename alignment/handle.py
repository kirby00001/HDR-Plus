import numpy as np


def downSample(fullSize):
    """
    Down sample full size image into a grayscale image by averaging 2x2 block
    :param fullSize: raw file in ndarray format
    :return: grayscale image
    """
    return fullSize[0::2, 0::2] + fullSize[1::2, 0::2] + fullSize[0::2, 1::2] + fullSize[1::2, 1::2] // 4


if __name__ == "__main__":
    raw = np.random.randint(low=1, high=255, size=(3000, 3000), dtype="uint8")
    gray = downSample(raw)
    print(raw)
    print(gray.shape)
    print(gray.dtype)
    print(gray)
