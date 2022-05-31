import rawpy

if __name__ == "__main__":
    with rawpy.imread("./bursts/0006_20160722_115157_431/payload_N001.dng") as raw:
        bayer_visible = raw.raw_image_visible
        print(bayer_visible.shape)
        print(bayer_visible)
