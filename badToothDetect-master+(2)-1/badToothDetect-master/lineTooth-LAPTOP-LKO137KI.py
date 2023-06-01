from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io,data_dir,filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

dirpath = 'toothLines'
newdir = 'Lines_tooth'

if not os.path.exists(newdir):
    os.makedirs(newdir)

order = [int(i.strip(".png")) for i in os.listdir(dirpath) if i.endswith(".png")]
jpglist = [f"{i}.png" for i in sorted(order)]  # direct reading of potentially non-sequential frames

radius = 1  # The Value of Range Radius in LBP Algorithm
n_points = 8 * radius # field pixel number
for i, png in enumerate(jpglist):
# settings for LBP
# 读取图像
    old = dirpath + f'/{png}'
    img = cv2.imread(old)  # return the numpy.array object return the array may not be available for the following acquisition thresholds
#Display to plt requires conversion from BGR to RGB. If cv2.imshow(win_name, image), no conversion is required


    #This is Sobel.

    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # Back to uint8
    absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)



    #This is Candy.
    """
    img = cv2.GaussianBlur(img,(3,3),0) #用高斯平滑处理原图像降噪。
    result = cv2.Canny(img, 20, 150) #最大最小阈值
    """


    #This is Laplace.

    """
    gray_lap = cv2.Laplacian(img,cv2.CV_16S,ksize = 3)
    result = cv2.convertScaleAbs(gray_lap)
    """

    # binarization
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # retval, result = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY)



    # In the middle is the algorithm.
    # cv2.imwrite(new, invert)
    new = newdir + f'/{png}'
    cv2.imwrite(new, result)

    print(f'{i + 1} / {len(jpglist)}')