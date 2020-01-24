import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


bf = cv2.BFMatcher()
# noinspection PyUnresolvedReferences
descriptor = cv2.xfeatures2d.SIFT_create()

video = cv2.VideoCapture(0)

old_frame = None
images = []
kernel = np.ones((5, 5), np.uint8)
T = None
while True:
    _, frame = video.read()
    height, width = frame.shape[:2]

    curr_vision = cv2.resize(frame, (width // 2, height // 2), cv2.INTER_AREA)

    if old_frame is None:
        T = np.float32([[1, 0, width/2-width/4], [0, 1, height/2-height/4]])
        old_frame = frame
        continue

    trainImg = frame.copy()

    queryImg = old_frame
    queryImg = cv2.resize(queryImg, (width // 2, height // 2), cv2.INTER_AREA)
    queryImg = cv2.warpAffine(queryImg, T, (width, height))

    kpsA, features1 = descriptor.detectAndCompute(trainImg, None)  # new
    kpsB, features2 = descriptor.detectAndCompute(queryImg, None)  # old

    matches = bf.knnMatch(features1, features2, k=2)
    best_matches = []
    for m, n in matches:
        if m.distance < 0.3 * n.distance:
            best_matches.append(m)

    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    ptsA = np.float32([kpsA[m.queryIdx] for m in best_matches])
    ptsB = np.float32([kpsB[m.trainIdx] for m in best_matches])
    try:
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4)
    except cv2.error:
        old_frame = frame
        continue
    try:
        result = cv2.warpPerspective(trainImg, H, (width, height))
    except cv2.error:
        continue
    images.append(result)

    output = None
    for image in images:
        gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY)
        mask2 = cv2.erode(mask2, kernel, iterations=1)
        result = cv2.bitwise_not(image)
        output = cv2.bitwise_not(result, output, mask=mask2)

    cv2.imshow('Key Frames Joined', output)
    cv2.imshow('Current Vision', curr_vision)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
video.read()
