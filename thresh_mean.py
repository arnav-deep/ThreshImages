import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt


def threshold_mean(image):
    return np.mean(image)


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

    binary_image = img > threshold_mean(img)
    plt.imshow(binary_image, cmap='gray')
    plt.show()
