import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

#images = [cv2.imread(f"Input/{name}", 0) for name in os.listdir("Input/") if name in ["NBR9201_04769.jpg", "NBR9201_04774.jpg"]]
images = [cv2.imread(f"Input/{name}", 0) for name in os.listdir("Input/")]

# Initialise ORB detector
orb = cv2.ORB_create(8000)
kps = []
dess = []

# Compute keypoints and descriptors for every image
for img in images:
    kp, des = orb.detectAndCompute(img, None)

    kps.append(kp)
    dess.append(des)


# Create feature matcher
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)

for i1, img1 in enumerate(images):
    print(f"i_1 = {i1}")  # Image nr 1 index

    for i2, img2 in enumerate(images):
        if i1 == i2:
            continue
        print(f"i_2 = {i2}")

        # Match descriptors
        matches = matcher.match(dess[i1], dess[i2], None)
        matches = sorted(matches, key=lambda x: x.distance)

        # Get matched keypoint coordinates
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kps[i1][match.queryIdx].pt
            points2[i, :] = kps[i2][match.trainIdx].pt

        # Get homography matrix between images
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

        img3 = cv2.drawMatches(img1, kps[i1], img2, kps[i2], matches[: 70], outImg=None, flags=2)

        plt.imshow(mask)
        plt.show()

        break
    break





