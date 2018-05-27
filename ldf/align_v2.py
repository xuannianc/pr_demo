"""
使用 SURF，提升速度
"""
import cv2
import numpy as np


def align(query_img, train_img, aligned_imgpath):
    # Load the images in gray scale
    img1 = cv2.imread(query_img, 0)
    img2 = cv2.imread(train_img, 0)

    # Detect the SIFT key points and compute the descriptors for the two images
    suft = cv2.xfeatures2d.SURF_create(hessianThreshold=1000)

    keyPoints1, descriptors1 = suft.detectAndCompute(img1, None)
    keyPoints2, descriptors2 = suft.detectAndCompute(img2, None)
    print('len of kp1={}'.format(len(keyPoints1)))
    print('len of kp2={}'.format(len(keyPoints2)))

    # Create brute-force matcher object
    bf = cv2.BFMatcher()

    # Match the descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    print('len of matches={}'.format(len(matches)))
    # Select the good matches using the ratio test
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)

    print('len of good matches={}'.format(len(goodMatches)))

    # Apply the homography transformation if we have enough good matches
    MIN_MATCH_COUNT = 1000

    if len(goodMatches) > MIN_MATCH_COUNT:
        # Get the good key points positions
        sourcePoints = np.float32([keyPoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
        destinationPoints = np.float32([keyPoints2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

        # Obtain the homography matrix
        M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)

        # Apply the perspective transformation to the source image corners
        h, w = img2.shape
        aligned = cv2.warpPerspective(img1, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        # cv2.imshow('aligned', aligned)
        print(M)
        cv2.imwrite(aligned_imgpath, aligned)
        # cv2.imwrite('aligned_' + query_img, aligned)
        # cv2.waitKey(0)


# align('sh-4.jpg', 'sh-0.jpg')
