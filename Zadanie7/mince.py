import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def print(img, title, p):
    plt.subplot(2, 2, p)
    plt.title(title)
    plt.tight_layout()
    plt.imshow(img, 'gray')


if __name__ == '__main__':
    img_original = cv.imread("obr/mince.png", 0)

    maxval = 255
    prah = 81

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (4, 4))

    _, img_prah2 = cv.threshold(img_original, prah, maxval, cv.THRESH_BINARY)
    img_prah3 = cv.morphologyEx(img_prah2, cv.MORPH_ELLIPSE, kernel)
    img_canny = cv.Canny(img_prah3, 100, 200)

    print(img_original, "Org pic", 1)
    print(img_prah2, "Mince - bin prah = %i" % prah, 2)
    print(img_prah3, "Mince - open", 3)
    print(img_canny, "Mince - Canny", 4)

    plt.savefig("mince.png")
    plt.show()
