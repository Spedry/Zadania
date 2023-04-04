import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img1 = cv.imread("obr/lena.jpg", 0)

    kernel1 = np.ones((3, 3), np.float32) / 9
    kernel2 = np.ones((15, 15), np.float32) / 225

    img2 = cv.filter2D(img1, -1, kernel1)
    img3 = cv.filter2D(img1, -1, kernel2)

    # Apply Gaussian blur with a 5x5 kernel and sigma of 0
    img_blur = cv.GaussianBlur(img1, (5, 5), 0)

    # Apply Canny edge detection with thresholds of 100 and 200
    img_edges = cv.Canny(img1, 100, 200)

    # Apply Sobel operator to compute the x and y gradients
    img_sobel_x = cv.Sobel(img1, cv.CV_64F, 1, 0, ksize=5)
    img_sobel_y = cv.Sobel(img1, cv.CV_64F, 0, 1, ksize=5)

    # Compute the Laplacian of the image
    img_laplacian = cv.Laplacian(img1, cv.CV_64F)

    # Apply adaptive threshold with a block size of 11 and a constant of 2
    img_adaptive_threshold = cv.adaptiveThreshold(img1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # Erode the foreground pixels with a 3x3 kernel
    kernel = np.ones((3, 3), np.uint8)
    img_eroded = cv.erode(img1, kernel, iterations=1)

    # Dilate the foreground pixels with a 3x3 kernel
    img_dilated = cv.dilate(img1, kernel, iterations=1)

    # Perform a closing operation with a 5x5 kernel
    img_closed = cv.morphologyEx(img1, cv.MORPH_CLOSE, kernel)

    # Display the images
    cv.imshow('Original', img1)
    cv.imshow('Blur', img_blur)
    cv.imshow('Edges', img_edges)
    cv.imshow('Sobel X', img_sobel_x)
    cv.imshow('Sobel Y', img_sobel_y)
    cv.imshow('Laplacian', img_laplacian)
    cv.imshow('Adaptive Threshold', img_adaptive_threshold)
    cv.imshow('Eroded', img_eroded)
    cv.imshow('Dilated', img_dilated)
    cv.imshow('Closed', img_closed)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow("Gaussian blur _moje_parametre", img2) # vykreslenie obrazku cez opencv



    cv.waitKey(0)