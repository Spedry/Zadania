import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


if __name__ == '__main__':
    img_original = cv.imread("obr/tvary.png", 0)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (10, 10))

    img1 = cv.dilate(img_original, kernel, iterations = 2)
    img2 = cv.erode(img_original, kernel, iterations = 2)

    cv.imshow("Org", img_original)
    cv.imshow("Dilate", img1)
    cv.imshow("Erode", img2)

    cv.waitKey(0)
    cv.destroyAllWindows()