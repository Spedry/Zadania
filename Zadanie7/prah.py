import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def print(img, title, p):
    plt.subplot(2, 2, p)
    plt.title(title)
    plt.tight_layout()
    plt.imshow(img, 'gray')


if __name__ == '__main__':
    img_original = cv.imread("obr/rampa2.png", 0)

    maxval = 255
    prahy = [20, 80, 150, 240]

    _, img_prah50 = cv.threshold(img_original, prahy[0], maxval, cv.THRESH_BINARY)
    _, img_prah100 = cv.threshold(img_original, prahy[1], maxval, cv.THRESH_BINARY)
    _, img_prah150 = cv.threshold(img_original, prahy[2], maxval, cv.THRESH_BINARY)
    _, img_prah200 = cv.threshold(img_original, prahy[3], maxval, cv.THRESH_BINARY)

    print(img_prah50, "prah = %i" % prahy[0], 1)
    print(img_prah100, "prah = %i" % prahy[1], 2)
    print(img_prah150, "prah = %i" % prahy[2], 3)
    print(img_prah200, "prah = %i" % prahy[3], 4)
    plt.savefig("prahy.png")
    #plt.show()