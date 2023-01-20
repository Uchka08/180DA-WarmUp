# Reference https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
#https://stackoverflow.com/questions/73808864/get-most-dominant-colors-from-video-opencv-python

import numpy as np
import cv2 as cv


cap = cv.VideoCapture(0)
n_clusters = 5

while True:
    status, image = cap.read()
    if not status:
        break

    # to reduce complexity resize the image
    data = cv.resize(image, (100, 100)).reshape(-1, 3)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)

    cluster_sizes = np.bincount(labels.flatten())

    palette = []
    for cluster_idx in np.argsort(-cluster_sizes):
        palette.append(np.full((image.shape[0], image.shape[1], 3), fill_value=centers[cluster_idx].astype(int), dtype=np.uint8))
    palette = np.hstack(palette)

    sf = image.shape[1] / palette.shape[1]
    out = np.vstack([image, cv.resize(palette, (0, 0), fx=sf, fy=sf)])

    cv.imshow("dominant_colors", out)
    cv.waitKey(1)