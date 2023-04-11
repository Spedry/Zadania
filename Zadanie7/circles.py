import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


if __name__ == '__main__':
    img_original = cv.imread("obr/circles.jpg", 0)

    #kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    #kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (10, 10))

    img1 = cv.morphologyEx(img_original, cv.MORPH_OPEN, kernel)
    img2 = cv.morphologyEx(img_original, cv.MORPH_CLOSE, kernel)

    cv.imshow("Circles", img_original)
    cv.imshow("Circles - open", img1)
    cv.imshow("Circles - close", img2)

    cv.waitKey(0)
    cv.destroyAllWindows()