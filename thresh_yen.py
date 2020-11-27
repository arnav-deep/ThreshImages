import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from skimage.exposure import histogram


def _validate_image_histogram(image, hist, nbins=None):
    if image is None and hist is None:
        raise Exception("Either image or hist must be provided.")

    if hist is not None:
        if isinstance(hist, (tuple, list)):
            counts, bin_centers = hist
        else:
            counts = hist
            bin_centers = np.arange(counts.size)
    else:
        counts, bin_centers = histogram(
            image.ravel(), nbins, source_range='image'
        )
    return counts.astype(float), bin_centers


def threshold_yen(image=None, nbins=256, *, hist=None):
    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # On blank images (e.g. filled with 0) with int dtype, `histogram()`
    # returns ``bin_centers`` containing only one value. Speed up with it.
    if bin_centers.size == 1:
        return bin_centers[0]

    # Calculate probability mass function
    pmf = counts.astype(np.float32) / counts.sum()
    P1 = np.cumsum(pmf)  # Cumulative normalized histogram
    P1_sq = np.cumsum(pmf ** 2)
    # Get cumsum calculated from end of squared array:
    P2_sq = np.cumsum(pmf[::-1] ** 2)[::-1]
    # P2_sq indexes is shifted +1. I assume, with P1[:-1] it's help avoid
    # '-inf' in crit. ImageJ Yen implementation replaces those values by zero.
    crit = np.log(((P1_sq[:-1] * P2_sq[1:]) ** -1) * (P1[:-1] * (1.0 - P1[:-1])) ** 2)
    return bin_centers[crit.argmax()]


if __name__ == '__main__':
    sysls = sys.argv
    imgfile = None
    ndim = 15
    img_method = 'gaussian'
    if len(sysls) == 2:
        imgfile = sysls[1]
    elif len(sysls) == 3:
        imgfile = sysls[1]
        ndim = int(sysls[2])

    # imgfile = 'cat.jpg'
    img = None
    try:
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        print("Add image to args or in the code.")
        exit()

    binary_image = img > threshold_yen(img, ndim)
    plt.imshow(binary_image, cmap='gray')
    plt.show()
