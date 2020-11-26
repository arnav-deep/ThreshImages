import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from scipy import ndimage as ndi


def l_mean(arr):
    return arr.mean()


def checkArray(array, ndim, arg_name='image'):
    array = np.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % arg_name)
    if array.ndim not in ndim:
        raise ValueError(msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim])))


def threshold_local(image, block_size, method='gaussian', offset=0,
                    mode='reflect', param=None, cval=0):
    if block_size % 2 == 0:
        raise ValueError("The kwarg ``block_size`` must be odd! Given "
                         "``block_size`` {0} is even.".format(block_size))
    checkArray(image, 2)
    thresh_image = np.zeros(image.shape, 'double')
    if method == 'generic':
        if param is None:
            param = l_mean
        ndi.generic_filter(image, param, block_size,
                           output=thresh_image, mode=mode, cval=cval)
    elif method == 'gaussian':
        if param is None:
            # automatically determine sigma which covers > 99% of distribution
            sigma = (block_size - 1) / 6.0
        else:
            sigma = param
        ndi.gaussian_filter(image, sigma, output=thresh_image, mode=mode,
                            cval=cval)
    elif method == 'mean':
        mask = 1. / block_size * np.ones((block_size,))
        # separation of filters to speedup convolution
        ndi.convolve1d(image, mask, axis=0, output=thresh_image, mode=mode,
                       cval=cval)
        ndi.convolve1d(thresh_image, mask, axis=1, output=thresh_image,
                       mode=mode, cval=cval)
    elif method == 'median':
        ndi.median_filter(image, block_size, output=thresh_image, mode=mode,
                          cval=cval)
    else:
        raise ValueError("Invalid method specified. Please use `generic`, "
                         "`gaussian`, `mean`, or `median`.")

    return thresh_image - offset


if __name__ == '__main__':
    sysls = sys.argv
    imgfile = None
    ndim = 15
    img_method = 'gaussian'
    if len(sysls) == 2:
        imgfile = sysls[1]
    elif len(sysls) == 3:
        imgfile = sysls[1]
        img_method = sysls[2]
    elif len(sysls) == 4:
        imgfile = sysls[1]
        img_method = sysls[2]
        ndim = int(sysls[3])

    # imgfile = 'cat.jpg'
    img = None
    try:
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        print("Add image to args or in the code.")
        exit()

    binary_image = img > threshold_local(img, ndim, img_method)
    plt.imshow(binary_image, cmap='gray')
    plt.show()
