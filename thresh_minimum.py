import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from skimage.exposure import histogram
from scipy import ndimage as ndi


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


def threshold_minimum(image=None, nbins=256, max_iter=10000, *, hist=None):

    def find_local_maxima_idx(hist):
        # We can't use scipy.signal.argrelmax
        # as it fails on plateaus
        maximum_idxs = list()
        direction = 1

        for i in range(hist.shape[0] - 1):
            if direction > 0:
                if hist[i + 1] < hist[i]:
                    direction = -1
                    maximum_idxs.append(i)
            else:
                if hist[i + 1] > hist[i]:
                    direction = 1

        return maximum_idxs

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    smooth_hist = counts.astype(np.float64, copy=False)

    for counter in range(max_iter):
        smooth_hist = ndi.uniform_filter1d(smooth_hist, 3)
        maximum_idxs = find_local_maxima_idx(smooth_hist)
        if len(maximum_idxs) < 3:
            break

    if len(maximum_idxs) != 2:
        raise RuntimeError('Unable to find two maxima in histogram')
    elif counter == max_iter - 1:
        raise RuntimeError('Maximum iteration reached for histogram'
                           'smoothing')

    threshold_idx = np.argmin(smooth_hist[maximum_idxs[0]:maximum_idxs[1] + 1])

    return bin_centers[maximum_idxs[0] + threshold_idx]


if __name__ == '__main__':
    sysls = sys.argv
    imgfile = None
    img = None
    if len(sysls) == 2:
        imgfile = sysls[1]

    # imgfile = 'cat.jpg'
    try:
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        print("Add image to args or in the code.")
        exit()

    binary_image = img > threshold_minimum(img)
    plt.imshow(binary_image, cmap='gray')
    plt.show()
