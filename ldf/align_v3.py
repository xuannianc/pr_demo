"""
使用 SURF，提升速度
"""
import cv2
import numpy as np
import os


def align(query_image_path, train_image_path):
    # Load the images in gray scale
    query_image = cv2.imread(query_image_path, 0)
    train_image = cv2.imread(train_image_path, 0)

    # Detect the SIFT key points and compute the descriptors for the two images
    suft = cv2.xfeatures2d.SURF_create(hessianThreshold=3000)

    keyPoints1, descriptors1 = suft.detectAndCompute(query_image, None)
    keyPoints2, descriptors2 = suft.detectAndCompute(train_image, None)
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
        h, w = train_image.shape
        aligned_gray = cv2.warpPerspective(query_image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        # print(M)
        # cv2.imwrite(aligned_imgpath, aligned)
        # cv2.imshow('aligned', aligned)
        # cv2.waitKey(0)
        aligned_image = cv2.cvtColor(aligned_gray, cv2.COLOR_GRAY2BGR)
        media_dir, query_image_name = os.path.split(query_image_path)
        cv2.imwrite(os.path.join(media_dir, 'aligned_' + query_image_name), aligned_image)
        return aligned_image

# align('sh-4.jpg', 'sh-0.jpg')
