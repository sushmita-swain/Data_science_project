import cv2
import numpy as np

img = cv2.imread('images/image1.jpeg')
res = cv2.resize(img, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)

# INTER_NEAREST - a nearest-neighbor interpolation
# INTER_LINEAR - a bilinear interpolation (used by default)
# INTER_AREA - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
# INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
# INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood