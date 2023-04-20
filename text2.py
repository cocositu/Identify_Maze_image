import cv2
import numpy as np
import time


cap = cv2.VideoCapture(0)  # 0 表示默认的摄像头
while True:
    # 读取一帧图像
    ret_1, img = cap.read()
    # 加载图像
    # img = cv2.imread("image2.jpg")

    # 进行图像处理操作
    # ...
    # 缩放图像大小
    img = cv2.resize(img, (640, 480))
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 对灰度图像进行二值化处理
    ret_2, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 对二值化图像进行腐蚀和膨胀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    # 高斯模糊
    blur = cv2.GaussianBlur(dilation, (5, 5), 0)
    # Canny边缘检测
    edges = cv2.Canny(blur, 50, 150)
    cv2.imshow('Edges', edges)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("gray", dilation)

    # 将图像转换为HSV颜色空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 定义红色和蓝色的HSV值范围
    lower_red = (0, 50, 50)
    upper_red = (10, 255, 255)
    lower_blue = (115, 50, 50)
    upper_blue = (125, 255, 255)
    # 创建掩膜以检测红色和蓝色矩形
    mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    # 消除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    # 查找轮廓
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制红色和蓝色矩形的轮廓
    img_contours = img.copy()
    # 创建掩膜图像
    mask = np.zeros_like(img_contours)
    # cv2.drawContours(img_contours, contours_red, -1, (0, 0, 255), 2)
    #cv2.drawContours(img_contours, contours_blue, -1, (255, 0, 0), 2)

    for contour in contours_red:
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        # 对轮廓进行逼近
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        if 3 <= len(approx) <= 6 and 300 <= w * h < 1500:
            cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for contour in contours_blue:
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        # 对轮廓进行逼近
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        if 3 <= len(approx) <= 6 and 300 <= w * h < 1500:
                cv2.rectangle(img_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # 遍历轮廓
    for contour in contours:
        # 画出轮廓
        #cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        # 对轮廓进行逼近
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        area = cv2.contourArea(approx)
        #cv2.drawContours(img_contours, approx, -1, (0, 255, 0), 2)
        if len(approx) > 5 and  80 < area < 200:
            radius = np.sqrt(area / np.pi)
            (x, y), _ = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            if radius != 0 and abs (1 - (perimeter / (2 * np.pi * radius))) <= 0.1:
                cv2.circle(img_contours, center, int(radius), (0, 0, 255), -1)


    # 显示图像
    cv2.imshow("img", img_contours)
    time.sleep(0.06)
    # 检测是否按下了 q 键，如果是则退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()




