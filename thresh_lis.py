import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt


def threshold_lis(image, *, tolerance=None, initial_guess=None, iter_callback=None):

    image = image[~np.isnan(image)]
    if image.size == 0:
        return np.nan

    if np.all(image == image.flat[0]):
        return image.flat[0]

    image = image[np.isfinite(image)]
    if image.size == 0:
        return 0.

    # Li's algorithm requires positive image (because of log(mean))
    image_min = np.min(image)
    image -= image_min
    tolerance = tolerance or np.min(np.diff(np.unique(image))) / 2

    # Initial estimate for iteration. See "initial_guess" in the parameter list
    if initial_guess is None:
        t_next = np.mean(image)
    elif callable(initial_guess):
        t_next = initial_guess(image)
    elif np.isscalar(initial_guess):  # convert to new, positive image range
        t_next = initial_guess - image_min
        image_max = np.max(image) + image_min
        if not 0 < t_next < np.max(image):
            msg = ('The initial guess for threshold_li must be within the '
                   'range of the image. Got {} for image min {} and max {} '
                   .format(initial_guess, image_min, image_max))
            raise ValueError(msg)
    else:
        raise TypeError('Incorrect type for `initial_guess`; should be '
                        'a floating point value, or a function mapping an '
                        'array to a floating point value.')

    # initial value for t_curr must be different from t_next by at
    # least the tolerance. Since the image is positive, we ensure this
    # by setting to a large-enough negative number
    t_curr = -2 * tolerance

    # Callback on initial iterations
    if iter_callback is not None:
        iter_callback(t_next + image_min)

    # Stop the iterations when the difference between the
    # new and old threshold values is less than the tolerance
    while abs(t_next - t_curr) > tolerance:
        t_curr = t_next
        foreground = (image > t_curr)
        mean_fore = np.mean(image[foreground])
        mean_back = np.mean(image[~foreground])

        t_next = ((mean_back - mean_fore) /
                  (np.log(mean_back) - np.log(mean_fore)))

        if iter_callback is not None:
            iter_callback(t_next + image_min)

    threshold = t_next + image_min
    return threshold


if __name__ == '__main__':
    sysls = sys.argv
    imgfile = None
    ndim = 15
    img_method = 'gaussian'
    if len(sysls) == 2:
        imgfile = sysls[1]

    # imgfile = 'cat.jpg'
    img = None
    try:
        img = cv2.imread(imgfile)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        print("Add image to args or in the code.")
        exit()

    binary_image = img > threshold_lis(img)
    plt.imshow(binary_image, cmap='gray')
    plt.show()
