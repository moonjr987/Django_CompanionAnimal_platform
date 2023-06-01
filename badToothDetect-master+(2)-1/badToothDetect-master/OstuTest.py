#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

#image = cv2.imread("/Users/JPEAC/OneDrive/바탕 화면/badToothDetect-master+(2)-1/badToothDetect-master/tooth/12.png")

img_path='C:/Users/PC/Desktop/badToothDetect-master+(2)-1/badToothDetect-master/tooth/12.png'
img_ar=np.fromfile(img_path,np.uint8)
img_path=cv2.imdecode(img_ar,cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img_path, cv2.COLOR_BGR2GRAY)

plt.subplot(131), plt.imshow(img_path, "gray")
plt.title("source image"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.hist(img_path.ravel(), 256)
plt.title("Histogram"), plt.xticks([]), plt.yticks([])
ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU) #방법선택THRESH_OTSU
plt.subplot(133), plt.imshow(th1, "gray")
plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
plt.show() 
