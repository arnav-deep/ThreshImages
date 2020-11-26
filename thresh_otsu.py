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


def threshold_otsu(image=None, nbins=256, *, hist=None):

    if image is not None and image.ndim > 2 and image.shape[-1] in (3, 4):
        msg = "Image has more than 1 channel."
        warn(msg.format(image.shape))

    if image is not None:
        first_pixel = image.ravel()[0]
        if np.all(image == first_pixel):
            return first_pixel

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]

    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    idx = np.argmax(variance12)
    threshold = bin_centers[idx]

    return threshold


if __name__ == '__main__':
    sysls = sys.argv
    imgfile = None
    nbin = 15
    img = None
    if len(sysls) == 2:
        imgfile = sysls[1]
    elif len(sysls) == 3:
        imgfile = sysls[1]
        nbin = sysls[2]

    # imgfile = 'cat.jpg'
    try:
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        print("Add image to args or in the code.")
        exit()

    binary_image = img > threshold_otsu(img, nbin)
    plt.imshow(binary_image, cmap='gray')
    plt.show()
