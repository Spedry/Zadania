import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def print_graf(img1, img2, img3):
    ##################################
    plt.subplot(2, 3, 1)
    plt.title("Lungs")
    plt.imshow(img1, 'gray')

    plt.tight_layout()
    plt.subplot(2, 3, 4)
    plt.hist(img1.ravel(), 50, [0, 256])  # ravel / konverzia 2D do 1D poľa
    ##################################
    plt.subplot(2, 3, 2)
    plt.title("Equalized - Lungs")
    plt.imshow(img2, 'gray')

    plt.tight_layout()
    plt.subplot(2, 3, 5)
    plt.hist(img2.ravel(), 50, [0, 256])  # ravel / konverzia 2D do 1D poľa
    ##################################
    plt.subplot(2, 3, 3)
    plt.title("CLAHE Lungs")
    plt.imshow(img3, 'gray')

    plt.subplot(2, 3, 6)
    plt.tight_layout()
    plt.hist(img3.ravel(), 200, [0, 256])  # ravel / konverzia 2D do 1D poľa
    ##################################

    plt.savefig("Lungs.png")
    plt.show()


if __name__ == '__main__':
    img1 = cv.imread("obr/pluca2.png", 0)
    img2 = cv.equalizeHist(img1)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    img3 = clahe.apply(img1)

    print_graf(img1, img2, img3)
