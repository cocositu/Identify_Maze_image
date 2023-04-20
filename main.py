import cv2 as cv
import numpy as np

# 加载图像
img = cv.imread('maze_img.jpg')


# 将图像转换为灰度图像
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 对灰度图像进行二值化处理
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# cv.imwrite('output4.jpg',thresh);
# 对二值化图像进行腐蚀和膨胀操作
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dilation = cv.dilate(thresh, kernel, iterations=1)
erosion = cv.erode(dilation, kernel, iterations=1)


erosion = cv.erode(dilation, kernel, iterations=12)
cv.imwrite('output3.jpg', erosion)

cv.waitKey(0)
cv.destroyAllWindows()
