import cv2

def nothing(x):


    pass
def Identifiy_Color(_img, lower_color, upper_color):
    # 将图像转换为HSV颜色空间
    img_hsv = cv2.cvtColor(_img, cv2.COLOR_BGR2HSV)
    # 创建掩膜以检测红色和蓝色矩形
    _mask_color = cv2.inRange(img_hsv, lower_color, upper_color)
    # 消除噪声
    _kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # _mask_color  = cv2.dilate(_mask_color , _kernel, iterations=3)
    # _mask_color  = cv2.erode(_mask_color , _kernel, iterations=2)
    _mask_color = cv2.morphologyEx(_mask_color, cv2.MORPH_OPEN, _kernel)
    cv2.imshow('cc',_mask_color)


def on_slider1(val):
    global slider1_val
    slider1_val = val
    if slider2_val < slider1_val:
        cv2.setTrackbarPos("slider2", "image", slider1_val)


def on_slider2(val):
    global slider2_val
    slider2_val = val
    if slider2_val < slider1_val:
        cv2.setTrackbarPos("slider1", "image", slider2_val)


def on_slider3(val):
    global slider3_val
    slider3_val = val
    if slider4_val < slider3_val:
        cv2.setTrackbarPos("slider4", "image", slider3_val)


def on_slider4(val):
    global slider4_val
    slider4_val = val
    if slider4_val < slider3_val:
        cv2.setTrackbarPos("slider3", "image", slider4_val)


def on_slider5(val):
    global slider5_val
    slider5_val = val
    if slider6_val < slider5_val:
        cv2.setTrackbarPos("slider6", "image", slider5_val)


def on_slider6(val):
    global slider6_val
    slider6_val = val
    if slider6_val < slider5_val:
        cv2.setTrackbarPos("slider5", "image", slider6_val)

# 创建窗口和滑动条
cv2.namedWindow("image")
cv2.moveWindow("image", 0, 0)

cv2.createTrackbar("slider1", "image", 0, 179, on_slider1)
cv2.createTrackbar("slider2", "image", 0, 179, on_slider2)
cv2.createTrackbar("slider3", "image", 0, 255, on_slider3)
cv2.createTrackbar("slider4", "image", 0, 255, on_slider4)
cv2.createTrackbar("slider5", "image", 0, 255, on_slider5)
cv2.createTrackbar("slider6", "image", 0, 255, on_slider6)

# 初始化滑动条值
slider1_val = slider2_val = slider3_val = 0
slider4_val = slider5_val = slider6_val = 0

cap = cv2.VideoCapture(0)  # 0 表示默认的摄像头
# 显示图像和滑动条
while True:

    # 读取一帧图像#读取图像,预处理图像
    _, img = cap.read()
    img_c = img.copy()
    img = cv2.resize(img, (640, 480))
    lc_blue = (slider1_val, slider3_val, slider5_val)
    uc_blue = (slider2_val, slider4_val, slider6_val)
    Identifiy_Color(img, lc_blue, uc_blue)
    cv2.imshow('ss', img_c)

    if cv2.waitKey(1) == ord('q'):
        break
# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()