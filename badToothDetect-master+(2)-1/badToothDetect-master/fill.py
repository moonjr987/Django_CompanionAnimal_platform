import numpy as np
import os
import cv2 as cv
from matplotlib import pyplot as plt
from skimage import measure, color
from skimage.measure import label

dirpath = '/Users/JPEAC/OneDrive/바탕 화면/badToothDetect-master+(2)-1/badToothDetect-master/tooth'
newdir = '/Users/JPEAC/OneDrive/바탕 화면/badToothDetect-master+(2)-1/badToothDetect-master/fill_tooth'

if not os.path.exists(newdir):
    os.makedirs(newdir)

order = [int(i.strip(".png")) for i in os.listdir(dirpath) if i.endswith(".png")]
jpglist = [f"{i}.png" for i in sorted(order)]  # 비순차 프레임 직접 읽기

for i, png in enumerate(jpglist):
    old = dirpath + f'/{png}'
    img = cv.imread(old)  # numpy.array개체를 반환합니다. 배열은 다음 획득 임계값에 사용할 수 없습니다.
    #이미지 읽어올 때 흑백으로 불러오기
    img2 = img.copy() #추가

    # 컬러 이미지를 흑백 이미지(GRAY)로 변환
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


    # 배열의 이치화


    # 대진법으로 자동 역치를 구하다
    # ret1, imgray = cv.threshold(imgray, 0, 255, cv.THRESH_OTSU)
    # 스스로 역치를 설정하다
    imgray[imgray < 100] = 0
    imgray[imgray >= 100] = 255

    # 연결 영역 추출



    # 알고리즘에 따라 임계값을 가져옵니다.

    # 임계값은 인접 영역의 평균값에서 가져옵니다.
    # imgray = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

    # 임계값은 인접한 영역의 가중치 합을 취하며 가중치는 가우스 창입니다.
    # imgray = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)



    # MASK 그림 범위를 제한하기 위해 원본 그림을 보정합니다
    mask = 255 - imgray

    #  Marker 그림 만들기
    marker = np.zeros_like(imgray)
    marker[0, :] = 255
    marker[-1, :] = 255
    marker[:, 0] = 255
    marker[:, -1] = 255

    # 다음은 형태학적 홀 충전입니다.
    # 형태학적 재구성 ksize 전달된 커널 크기
    # 커널 모양 직사각형
    # SE = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(3, 3))
    # 커널 형상 교차형
    SE = cv.getStructuringElement(shape=cv.MORPH_CROSS, ksize=(3, 3))
    # 커널 모양 타원
    # SE = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(3, 3))
    while True:
        marker_pre = marker
        dilation = cv.dilate(marker, kernel=SE)
        marker = np.min((dilation, mask), axis=0)
        if (marker_pre == marker).all():
            break
    dst = 255 - marker
    filling = dst - imgray

    """
    # 연결 도메인 추출 수행
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(filling, connectivity=4)

    # 반환 값 보기
    # 연결역수
    print('num_labels = ', num_labels)
    # 연결 영역의 정보: 각 프로파일에 대응하는 x, y, width, height 및 면적
    print('stats = ', stats)
    # 연결 영역의 중심점
    print('centroids = ', centroids)
    # 각 픽셀의 라벨은 1, 2, 3.…이고 동일한 연결 도메인의 라벨은 동일합니다.
    print('labels = ', labels)

    # 서로 다른 연결 영역은 서로 다른 색을 부여한다
    output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    for i in range(1, num_labels):
        mask = labels == i
        output[:, :, 0][mask] = np.random.randint(0, 255)
        output[:, :, 1][mask] = np.random.randint(0, 255)
        output[:, :, 2][mask] = np.random.randint(0, 255)
    """



    # 반색 처리를 하다
    # filling = 255 - filling
    # 형태학 커널
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

    # 먼저 계산을 닫았다가 다시 켜다.
    # filling = cv.morphologyEx(filling, cv.MORPH_CLOSE, kernel, 1)
    # 형태학적 처리: 오픈 연산
    filling = cv.morphologyEx(filling, cv.MORPH_OPEN, kernel, 1)
    #


    # 이진화된 윤곽을 그립니다
    # contours, hierarchy = cv.findContours(filling, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 가장 바깥 실루엣만
    contours, hierarchy = cv.findContours(filling, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contour2, hierarchy = cv.findContours(filling, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)#추가
    #최대의 윤곽을 선택하여 채운다
    area = []
    for k in range(len(contours)):
        area.append(cv.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))

    
    #hull = cv2.convexHull(cnt)
    # 마지막은 라인의 두께감
    cv.drawContours(img, contours, max_idx, (212, 255,127 ), 1)

    #추가
    for idx, cont in enumerate(contour2): 
    # 랜덤한 컬러 추출 ---⑦
        color = [int(i) for i in np.random.randint(0,255, 3)]
    # 컨투어 인덱스 마다 랜덤한 색상으로 그리기 ---⑧
        cv.drawContours(img2, contour2, idx, color, 3)


    # 나타내다
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
    cv.imwrite(new, img2)
    print(f'{i + 1} / {len(jpglist)}')
