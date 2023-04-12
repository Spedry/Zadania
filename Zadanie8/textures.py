import numpy as np
import cv2 as cv
import math
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage
import matplotlib.pyplot as plt


if __name__ == '__main__':
    image2 = cv.imread('../Zadanie7/obr/latka.jpg')
    image2_gs = cv.imread('../Zadanie7/obr/latka.jpg', 0)

    # Calculating and displaying the entropy of the grayscale image
    image2_ent = entropy(image2_gs, disk(5))
    plt.imshow(image2_ent, cmap='gray')
    plt.show()

    # Applying threshold to the entropy image and displaying the result
    (T, thresh) = cv.threshold(image2_ent, 5.5, 255, cv.THRESH_BINARY)
    plt.imshow(thresh, cmap='gray')
    plt.show()

    # Applying morphological operations to the thresholded image to remove noise
    image2_morf = ndimage.binary_opening(thresh, structure=np.ones((4, 8))).astype(int)
    image2_morf = ndimage.binary_closing(image2_morf, structure=np.ones((13, 10))).astype(int)
    image2_morf = ndimage.binary_opening(image2_morf, structure=np.ones((12, 15))).astype(int)
    image2_morf = ndimage.binary_closing(image2_morf, structure=np.ones((16, 15))).astype(int)
    plt.imshow(image2_morf, cmap='gray')
    plt.show()

    # Creating masks based on the morphed image to separate the objects and their backgrounds
    mask = np.zeros(image2_morf.shape, dtype=np.uint8)
    mask = image2_morf.astype(np.uint8)
    mask2 = mask ^ 1
    image2_mask = cv.bitwise_and(image2_gs, image2_gs, mask=mask)
    image2_mask2 = cv.bitwise_and(image2_gs, image2_gs, mask=mask2)
    plt.imshow(image2_mask, cmap='gray')
    plt.show()
    plt.imshow(image2_mask2, cmap='gray')
    plt.show()

    # Finding the contours in the morphed image and drawing them on the original image
    contours, hierarchy = cv.findContours(image2_morf.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    image2_contours = cv.drawContours(image2, contours, -1, (0, 255, 0), 2)
    plt.imshow(image2_contours)
    plt.show()