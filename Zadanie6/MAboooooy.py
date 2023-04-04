import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def print_graf(img1, img2, img3):
    ##################################
    plt.subplot(2, 3, 1)
    plt.title("Boy")
    plt.imshow(img1, 'gray')

    plt.tight_layout()
    plt.subplot(2, 3, 4)
    plt.hist(img1.ravel(), 255, [0, 256])  # ravel / konverzia 2D do 1D poľa
    ##################################
    plt.subplot(2, 3, 2)
    plt.title("Equalized - BOY")
    plt.imshow(img2, 'gray')

    plt.tight_layout()
    plt.subplot(2, 3, 5)
    plt.hist(img2.ravel(), 255, [0, 256])  # ravel / konverzia 2D do 1D poľa
    ##################################
    plt.subplot(2, 3, 3)
    plt.title("CLAHE BOY")
    plt.imshow(img3, 'gray')

    plt.tight_layout()
    plt.subplot(2, 3, 6)
    plt.hist(img3.ravel(), 255, [0, 256])  # ravel / konverzia 2D do 1D poľa
    ##################################

    plt.savefig("Boy.png")
    plt.show()


def print_clahe(img1, img2):
    ##################################
    plt.subplot(2, 2, 1)
    plt.title("Adaptive1")
    plt.imshow(img1, 'gray')

    plt.tight_layout()
    plt.subplot(2, 2, 3)
    plt.hist(img1.ravel(), 255, [0, 256])  # ravel / konverzia 2D do 1D poľa
    ##################################
    plt.subplot(2, 2, 2)
    plt.title("Adaptive2")
    plt.imshow(img2, 'gray')

    plt.tight_layout()
    plt.subplot(2, 2, 4)
    plt.hist(img2.ravel(), 255, [0, 256])  # ravel / konverzia 2D do 1D poľa

    plt.savefig("Boy_clahe.png")
    plt.show()


if __name__ == '__main__':
    img1 = cv.imread("obr/chlapec2.png", 0)
    img2 = cv.equalizeHist(img1)

    clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(3, 3))

    img3 = clahe.apply(img1)

    print_graf(img1, img2, img3)

    clahe2 = cv.createCLAHE(clipLimit=1.5, tileGridSize=(20, 20))

    img4 = clahe2.apply(img1)

    print_clahe(img3, img4)
