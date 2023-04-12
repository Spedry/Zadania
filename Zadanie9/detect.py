import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


if __name__ == '__main__':
    training_original = cv.imread('../Zadanie7/obr/stopka.jpg', 0)
    training_brgTOrgb = cv.cvtColor(training_original, cv.COLOR_BGR2RGB)

    test_original = cv.imread('stop.jpg', 0)
    test_brgTOrgb = cv.cvtColor(test_original, cv.COLOR_BGR2RGB)


    test_brgTOrgb = cv.pyrDown(test_brgTOrgb)
    test_brgTOrgb = cv.pyrDown(test_brgTOrgb)

    #num_rows, num_cols = test_rgb.shape[:2]
    #rotation_matrix = cv.getRotationMatrix2D((num_cols / 2, num_rows / 2), 30, 1)
    #test_image = cv.warpAffine(test_image, rotation_matrix, (num_cols, num_rows))

    #test_gray = cv.cvtColor(test_rgb, cv.COLOR_RGB2GRAY)
    #training_gray = cv.cvtColor(training_rgb, cv.COLOR_RGB2GRAY)

    fx, plots = plt.subplots(1, 2, figsize=(20, 10))
    plots[0].set_title("Training Image")
    plots[0].imshow(training_brgTOrgb)
    plots[1].set_title("Testing Image")
    plots[1].imshow(test_brgTOrgb)
    plt.show()
    #########################################################
    sift = cv.SIFT_create(800)

    train_keypoints, train_descriptor = sift.detectAndCompute(training_original, None)

    test_keypoints, test_descriptor = sift.detectAndCompute(test_original, None)
    #########################################################
    keypoints_without_size = np.copy(training_brgTOrgb)
    keypoints_with_size = np.copy(training_brgTOrgb)

    cv.drawKeypoints(training_brgTOrgb, train_keypoints, keypoints_without_size, color=(0, 255, 0))
    cv.drawKeypoints(training_brgTOrgb, train_keypoints, keypoints_with_size, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fx, plots = plt.subplots(1, 2, figsize=(20, 10))
    plots[0].set_title("Train keypoints With Size")
    plots[0].imshow(keypoints_with_size, cmap='gray')
    plots[1].set_title("Train keypoints Without Size")
    plots[1].imshow(keypoints_without_size, cmap='gray')
    plt.show()

    print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))
    print("Number of Keypoints Detected In The TestingImage: ", len(test_keypoints))
    ##################################################
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)

    matches = bf.match(train_descriptor, test_descriptor)

    matches=sorted(matches, key=lambda x: x.distance)

    result = cv.drawMatches(training_brgTOrgb, train_keypoints, test_brgTOrgb, test_keypoints, matches, test_brgTOrgb, flags=2)
    plt.figure()
    plt.rcParams['figure.figsize'] = [14.0, 7.0]
    plt.title('Best Matching Points')
    plt.imshow(result)
    plt.show()
    print("\nNumber of Matching Keypoints Between The Training and Testing Images: ", len(matches))
    ###################################################
    good_matches = matches[:25]
    src_pts = np.float32([train_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = training_brgTOrgb.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    dst += (w, 0)

    draw_params = dict(matchColor=(0, 255, 0), singlePointColor =None,matchesMask =matchesMask, flags =2)

    img3 = cv.drawMatches(training_original, train_keypoints, test_original, test_keypoints, good_matches, None, **draw_params)

    img3 = cv.polylines(img3, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
    plt.figure()
    plt.title('Found Object')
    plt.imshow(img3)
    plt.show()