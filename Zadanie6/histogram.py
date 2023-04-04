import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def print_pic():
    plt.imshow(img1, 'gray')
    plt.show()


def print_graf(img1, img2, ramp):
    ##################################
    plt.subplot(2, 3, 1)
    plt.title("Square")
    plt.imshow(img1, 'gray')

    plt.tight_layout()
    plt.subplot(2, 3, 4)
    plt.hist(img1.ravel(), 50, [0, 256])  # ravel / konverzia 2D do 1D poľa
    ##################################
    plt.subplot(2, 3, 2)
    plt.title("Square - flip")
    plt.imshow(img2, 'gray')

    plt.tight_layout()
    plt.subplot(2, 3, 5)
    plt.hist(img2.ravel(), 50, [0, 256])  # ravel / konverzia 2D do 1D poľa
    ##################################
    plt.subplot(2, 3, 3)
    plt.title("Ramp")
    plt.imshow(ramp, 'gray')

    plt.tight_layout()
    plt.subplot(2, 3, 6)
    plt.hist(ramp.ravel(), 200, [0, 256])  # ravel / konverzia 2D do 1D poľa
    ##################################

    plt.savefig("Histogram.png")
    plt.show()


if __name__ == '__main__':
    img1 = np.zeros((200, 200), np.uint8)

    # print_pic()

    cv.rectangle(img1, (0, 100), (200, 200), 255, -1)  # white
    cv.rectangle(img1, (0, 50), (150, 150), 200, -1)  # light
    cv.rectangle(img1, (0, 25), (125, 75), 100, -1)  #
    cv.rectangle(img1, (0, 50), (50, 200), 35, -1)  # dark

    # print_pic()

    img2 = np.flip(img1)

    print_graf(img1, img2, cv.imread("obr/rampa.png"))
