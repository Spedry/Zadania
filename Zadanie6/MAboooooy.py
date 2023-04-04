import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def print_graf(img1, img2, img3):
    ##################################
    # plot the original image in the first subplot
    plt.subplot(2, 3, 1)
    plt.title("Boy")
    plt.imshow(img1, 'gray')

    # plot the histogram of the original image in the fourth subplot
    plt.tight_layout()
    plt.subplot(2, 3, 4)
    plt.hist(img1.ravel(), 255, [0, 256])  # ravel / konverzia 2D do 1D poľa

    ##################################
    # plot the equalized image in the second subplot
    plt.subplot(2, 3, 2)
    plt.title("Equalized - BOY")
    plt.imshow(img2, 'gray')

    # plot the histogram of the equalized image in the fifth subplot
    plt.tight_layout()
    plt.subplot(2, 3, 5)
    plt.hist(img2.ravel(), 255, [0, 256])  # ravel / konverzia 2D do 1D poľa

    ##################################
    # plot the CLAHE image in the third subplot
    plt.subplot(2, 3, 3)
    plt.title("CLAHE BOY")
    plt.imshow(img3, 'gray')

    # plot the histogram of the CLAHE image in the sixth subplot
    plt.tight_layout()
    plt.subplot(2, 3, 6)
    plt.hist(img3.ravel(), 255, [0, 256])  # ravel / konverzia 2D do 1D poľa
    ##################################

    # save the plot as an image file and show the plot
    plt.savefig("Boy.png")
    plt.show()

def print_clahe(img1, img2):
    ##################################
    # plot the first CLAHE image in the first subplot
    plt.subplot(2, 2, 1)
    plt.title("Adaptive1")
    plt.imshow(img1, 'gray')

    # plot the histogram of the first CLAHE image in the third subplot
    plt.tight_layout()
    plt.subplot(2, 2, 3)
    plt.hist(img1.ravel(), 255, [0, 256])  # ravel / konverzia 2D do 1D poľa

    ##################################
    # plot the second CLAHE image in the second subplot
    plt.subplot(2, 2, 2)
    plt.title("Adaptive2")
    plt.imshow(img2, 'gray')

    # plot the histogram of the second CLAHE image in the fourth subplot
    plt.tight_layout()
    plt.subplot(2, 2, 4)
    plt.hist(img2.ravel(), 255, [0, 256])  # ravel / konverzia 2D do 1D poľa

    # save the plot as an image file and show the plot
    plt.savefig("Boy_clahe.png")
    plt.show()

if __name__ == '__main__':
    # read the input image
    img1 = cv.imread("obr/chlapec2.png", 0)

    # apply histogram equalization to the input image
    img2 = cv.equalizeHist(img1)

    # create a CLAHE object with a clip limit of 1.5 and a tile size of (3,3)
    clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(3, 3))

    img3 = clahe.apply(img1)

    print_graf(img1, img2, img3)

    clahe2 = cv.createCLAHE(clipLimit=1.5, tileGridSize=(20, 20))

    img4 = clahe2.apply(img1)

    print_clahe(img3, img4)