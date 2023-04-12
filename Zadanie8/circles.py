import numpy as np
import cv2 as cv
import math

points_arr_x = []
points_arr_y = []
point_no = 0


def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        points_arr_x.append(x)
        points_arr_y.append(y)
        print("x: " + str(x), ' ', "y: " + str(y))
        global point_no
        if point_no == 0:
            point_no += 1
        else:
            cv.line(image_rgb, (points_arr_x[len(points_arr_x) - 2], points_arr_y[len(points_arr_y) - 2]),
                    (points_arr_x[len(points_arr_x) - 1], points_arr_y[len(points_arr_y) - 1]), (0, 0, 0), 2, 8)
            print("Priemer: " + str(math.sqrt(
                (points_arr_x[len(points_arr_x) - 2] - points_arr_x[len(points_arr_x) - 1]) ** 2 + (
                            points_arr_y[len(points_arr_y) - 2] - points_arr_y[len(points_arr_y) - 1]) ** 2)))
            point_no = 0
            cv.imshow('image', image_rgb)


if __name__ == '__main__':
    # image_rgb = cv.imread('../Zadanie7/obr/coloredChips.png')
    image_rgb = cv.imread('obr3.jpg')
    cv.imshow('image', image_rgb)
    cv.setMouseCallback('image', click_event)
    cv.waitKey(0)

    # image_gs = cv.imread('../Zadanie7/obr/coloredChips.png', 0)
    image_gs = cv.imread('obr3.jpg', 0)
    circles = cv.HoughCircles(image_gs, cv.HOUGH_GRADIENT, 1, 80, param1=123, param2=25, minRadius=20, maxRadius=85)
    circles = np.uint16(np.around(circles))

    for i in circles[0, :]:
        cv.circle(image_rgb, (i[0], i[1]), i[2], (0, 0, 0), 2)
    cv.imshow('detected circles', image_rgb)
    cv.waitKey(0)
    cv.destroyAllWindows()
