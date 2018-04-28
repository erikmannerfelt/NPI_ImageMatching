import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

#images = [cv2.imread(f"Input/{name}", 0) for name in os.listdir("Input/") if name in ["NBR9201_04769.jpg", "NBR9201_04774.jpg"]]
images = [cv2.imread(f"Input/{name}", 0) for name in os.listdir("Input/")]

record = 0
count = 0
for i1, img1 in enumerate(images):
    break
    for i2, img2 in enumerate(images):
        if i1 != i2:
            orb = cv2.ORB_create(nfeatures=8000)

            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            print("Mean", np.mean([m.distance for m in matches][:100]))

            amount = len(matches)
            if amount > record:
                record = amount
                print(f"Record: {record}")
                img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

                plt.imshow(img3)
                plt.show()


            count += 1
            if count % 10 == 0:
                print(count)

kps = []
dess = []
orb = cv2.ORB_create(8000)
count = 0
for img in images:
    kp, des = orb.detectAndCompute(img, None)

    kps.append(kp)
    dess.append(dess)

    count += 1
    print(count)

