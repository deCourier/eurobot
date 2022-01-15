import cv2
from cv2 import aruco
import numpy as np
import rospy


def crop(frame):
    frame_zero = np.zeros(frame.shape, dtype=np.uint8)
    frame_zero[455:, 415:2020] = frame[455:, 415:2020]
    return frame_zero


def siftFeatures(frame, path):
    MIN_MATCH_COUNT = 50
    template = cv2.imread(path)

    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    template = cv2.resize(template, (0, 0), fx=0.5, fy=0.5)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()

    (keypoints1, descriptors1) = sift.detectAndCompute(frame_gray, None)
    (keypoints2, descriptors2) = sift.detectAndCompute(template_gray, None)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=200)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good = []
    for m, n in matches:
        img2_idx = m.trainIdx
        img1_idx = m.queryIdx
        pt1 = np.array(keypoints1[img1_idx].pt)
        pt2 = np.array(keypoints2[img2_idx].pt)
        dist = np.linalg.norm(pt1-pt2)
        if m.distance < 0.7 * n.distance and dist < 150:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        src_pts *= 2
        dst_pts *= 2

        matrix, mask = cv2.findHomography(src_pts, dst_pts)
        matches_mask = mask.ravel().tolist()

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matches_mask = None

    draw_params = dict(singlePointColor=None,
                       matchesMask=matches_mask,
                       flags=2)

    matches_img = cv2.drawMatches(frame_gray, keypoints1, template_gray, keypoints2, good, None, **draw_params)
    cv2.imwrite("../matches.jpg", matches_img)

    rospy.loginfo('Matches = {}'.format(len(good)))

    return matrix