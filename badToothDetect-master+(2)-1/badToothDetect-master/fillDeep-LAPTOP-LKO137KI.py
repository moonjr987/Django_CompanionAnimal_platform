import numpy as np
import os
import cv2 as cv
from matplotlib import pyplot as plt
dirpath = 'toothDeep'
newdir = 'fill_toothDeep'

if not os.path.exists(newdir):
    os.makedirs(newdir)

order = [int(i.strip(".png")) for i in os.listdir(dirpath) if i.endswith(".png")]
jpglist = [f"{i}.png" for i in sorted(order)]  # direct reading of potentially non-sequential frames

for i, png in enumerate(jpglist):
    old = dirpath + f'/{png}'
    img = cv.imread(old)  # And what's coming back is numpy.array The object returns an array that may not be used for the following acquisition thresholds



    # convert an image
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    # binarization of arrays

    # set one's own threshold
    imgray[imgray < 100] = 0
    imgray[imgray >= 100] = 255

    # algorithmic threshold acquisition

    # The threshold is taken from the average of adjacent areas.
    # imgray = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

    # The weighted sum of the adjacent regions of a threshold, weighted as a Gaussian window。
    # imgray = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)


    # The hole is filled below

    # MASK Image Obtained by Original Image Complementation
    mask = 255 - imgray

    # Construct Marker Images
    marker = np.zeros_like(imgray)
    marker[0, :] = 255
    marker[-1, :] = 255
    marker[:, 0] = 255
    marker[:, -1] = 255
    marker_0 = marker.copy()

   # Morphological reconstruction ksize is the passing kernel size
    # kernel shape rectangle
    # SE = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(3, 3))
    # intersection of core shapes
    SE = cv.getStructuringElement(shape=cv.MORPH_CROSS, ksize=(3, 3))
    # inner ellipse
    # SE = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3, 3))
    while True:
        marker_pre = marker
        dilation = cv.dilate(marker, kernel=SE)
        marker = np.min((dilation, mask), axis=0)
        if (marker_pre == marker).all():
            break
    dst = 255 - marker
    filling = dst - imgray

    # reverse color processing
    # filling = 255 - filling
    # morphological kernel
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # closed operation before opening
    # filling = cv.morphologyEx(filling, cv.MORPH_CLOSE, kernel, 1)
    # Morphological processing: open arithmetic
    filling = cv.morphologyEx(filling, cv.MORPH_OPEN, kernel, 1)
    #

    # Draw the resulting binarized outline.
    # contours, hierarchy = cv.findContours(filling, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # As long as it's the outermost profile.
    contours, hierarchy = cv.findContours(filling, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # select the maximum profile to fill
    area = []
    for k in range(len(contours)):
        area.append(cv.contourArea(contours[k]))
    if area!=[]:
        max_idx = np.argmax(np.array(area))



    # And the last one is line thickness.
    cv.drawContours(img, contours, max_idx, (212, 255, 127), 1)

    # display
    # plt.figure(figsize=(12, 6))  # width * height
    # plt.subplot(2, 3, 1), plt.imshow(imgray, cmap='gray'), plt.title('src'), plt.axis("off")
    # plt.subplot(2, 3, 2), plt.imshow(mask, cmap='gray'), plt.title('Mask'), plt.axis("off")
    # # plt.subplot(2, 3, 3), plt.imshow(marker_0, cmap='gray'), plt.title('Marker 0'), plt.axis("off")
    # plt.subplot(2, 3, 3), plt.imshow(img, cmap='gray'), plt.title('Marker 0'), plt.axis("off")
    # plt.subplot(2, 3, 4), plt.imshow(marker, cmap='gray'), plt.title('Marker'), plt.axis("off")
    # plt.subplot(2, 3, 5), plt.imshow(dst, cmap='gray'), plt.title('dst'), plt.axis("off")
    # plt.subplot(2, 3, 6), plt.imshow(filling, cmap='gray'), plt.title('Holes'), plt.axis("off")



    new = newdir + f'/{png}'
    # cv2.imwrite(new, invert)
    cv.imwrite(new, img)
    print(f'{i + 1} / {len(jpglist)}')
