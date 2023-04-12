import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def print(img, title, p):
    plt.subplot(2, 2, p)
    plt.title(title)
    plt.tight_layout()
    plt.imshow(img, 'gray')


def print_color(img, title, p):
    plt.subplot(2, 2, p)
    plt.title(title)
    plt.tight_layout()
    plt.imshow(img)


if __name__ == '__main__':
    img_original = cv.imread("obr/predmety.png", 0)
    img_original_color = cv.imread("obr/predmety.png", cv.IMREAD_COLOR)

    maxval = 255
    prahy = [120, 130]

    _, img_prah2 = cv.threshold(img_original_color, prahy[0], maxval, cv.THRESH_BINARY)
    _, img_prah3 = cv.threshold(img_original, prahy[1], maxval, cv.THRESH_BINARY)
    prah, img_prah4 = cv.threshold(img_original, 0, maxval, cv.THRESH_BINARY | cv.THRESH_OTSU)

    print_color(img_original_color, "org pic", 1)
    print_color(img_prah2, "color prah = %i" % prahy[0], 2)
    print(img_prah3, "optimal prah = %i" % prahy[1], 3)
    print(img_prah4, "OTSU = %i" % prah, 4)
    plt.savefig("predmety.png")
    plt.show()
