import cv2
from matplotlib.dviread import Box
import numpy as np
import copy


# Version : 0.4
# Time    : 2023.04.26.1.52
# Note    : 对drawMaze函数进行修改.除了drawImage以外所有的函数都不应
#           该在函数体里面直接更改img,drawImag有点问题


# Version : 0.3
# Time    : 2023.04.25.23.55
# Note    :本次将将迷宫转化为 二维数组的代码需要封装为函数。
#

# Version : 0.2
# Time    : 2023.04.25.16.53
# Note    :本次将函数注释完成，部分函数有些问题不应该在函数内更改原图像，同时主函数中将迷宫转化为
#         二维数组的代码需要封装为函数，有时候程序会莫名其妙崩溃，有可能是应为列表为空后使用cv函数
#         调用发生崩溃，后期将所有函数放到一个包内。


def PreDispose_Img(_img):
    """
        @brief 预处理图片，对原图像进行腐蚀膨胀，转化为灰度，二值化， 边缘检测以及寻找轮廓
        @param
            :param1 image 要处理的图片
        @return
            :val1   _contours  轮廓
            :val2   _hierarchy 轮廓层次结构
            :val3   _thresh    二值化图
            :val4   _edges     边缘检测图
        @note              备注
        @raises            异常
    """
    _gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
    _, _thresh = cv2.threshold(_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 消除噪声
    _kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    _thresh = cv2.morphologyEx(_thresh, cv2.MORPH_OPEN, _kernel)
    _thresh = cv2.morphologyEx(_thresh, cv2.MORPH_OPEN, _kernel)
    # 高斯模糊
    _blur = cv2.GaussianBlur(_thresh, (5, 5), 0)
    # Canny边缘检测
    _edges = cv2.Canny(_blur, 50, 150)
    # 寻找轮廓和层次结构
    _contours, _hierarchy = cv2.findContours(_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('th', _thresh)
    return _contours, _thresh, _hierarchy, _edges


def Cmp_CentPos(_contours):
    """
        @brief 求轮廓的中心点坐标
        @param
            :param1 _contours  一个图形的轮廓
        @return
            :val1   _cx         轮廓中心点x坐标
            :val2   _cy         轮廓中心点y坐标
        @note   使用一阶矩除以面积得到中心点坐标
        @raises            异常
    """
    M = cv2.moments(_contours)
    _cx = int(M['m10'] / M['m00'])
    _cy = int(M['m01'] / M['m00'])
    return _cx, _cy


def Sort_PositPoi(pos_arr):
    """
        @brief 重新排序定位点坐标
        @param
            :param1 pos_arr    定位点坐标数组
        @return
            :val1   points_arr 顺时针排序后的定位点坐标数组
        @note
        @raises            异常
    """
    s = []
    diff = []
    for _i in range(4):
        s.append(pos_arr[_i][0] + pos_arr[_i][1])
        diff.append(pos_arr[_i][0] - pos_arr[_i][1])
    # 对矩形框的顶点坐标按照顺时针方向排序
    s_min = np.argmin(s)
    d_min = np.argmin(diff)
    s_max = np.argmax(s)
    d_max = np.argmax(diff)
    points_arr = np.array([pos_arr[s_min], pos_arr[d_min], pos_arr[s_max], pos_arr[d_max]])
    return points_arr


##这个函数需要修改，传入的原图像尽量不要去修改
def Identify_LocalPoi(_img, _contours, _hierarchy):
    """
        @brief  在图像中找出四个定位点
        @param
            :param1 原图像
            :param2 图像中所有轮廓
            :param3 轮廓的层次结构
        @return
            :val1   标注好四个定位后的图像
            :val2   四个定位点的中心坐标
        @note
        @raises            异常
    """
    # LocPo_Num = 0; PosPoi = []; Loc_Box = []
    LocPo_Num, PosPoi,Loc_Box = 0, [], []
    for _i in range(len(_contours)):
        # d1一级嵌套， d2二级嵌套
        d1 = _hierarchy[0][_i][2]
        d2 = _hierarchy[0][d1][2]

        _perimeter = cv2.arcLength(_contours[_i], True)
        _approx = cv2.approxPolyDP(_contours[_i], 0.05 * _perimeter, True)
        _area = cv2.contourArea(_approx)

        if 300 < _area < 1500 and d1 != -1 and d2 != -1:
            LocPo_Num += 1
            # 获取回字定位图案的最小外接矩形
            _rect = cv2.minAreaRect(_contours[_i])
            # _box = np.int32(cv2.boxPoints(_rect))
            # 修复关键行：四舍五入并转换为整数
            _box = cv2.boxPoints(_rect)
            _box = np.round(_box).astype(np.int32)

            # 画出回字定位图案和定位点中心
            cx, cy =  Cmp_CentPos(_contours[_i])
            cv2.drawContours(_img, [_box], 0, (0, 0, 255), 2)
            # cv2.circle(_img, (cx, cy), 3, (0, 255, 0), -1)
            # 定位点坐标
            PosPoi.append([cx, cy])
            Loc_Box.append([_box])

    return  PosPoi, Loc_Box, LocPo_Num


def Transf_Image(_img, img_point):
    """
        @brief  通过四个定位点坐标去矫正图像
        @param
            :param1  原图像
            :param2  四个定位点坐标
        @return
            :val1
        @note       使用透视矩阵变换，图片转换成500x500像素大小的图片，
                    四个定位点坐标分别为[30, 30], [30, 446], [446, 446], [446, 30]
        @raises            异常
    """
    # 设置矩形的四个角点坐标
    src_points = np.float32(img_point)
    # 计算矩形的理论形状（假设为正方形）
    dst_points = np.float32([[30, 30], [30, 449], [449, 449], [449, 30]])
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # 进行透视变换，边缘使用白色填充
    rec_img = cv2.warpPerspective(_img, M, (479, 479), borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])
    return rec_img


def Recognize_Dot(_img, _contours):
    """
        @brief 找出迷宫外轮廓，八个宝藏的轮廓和坐标
        @param
            :param1 原图像
            :param2 图像的所有轮廓
        @return
            :val1   迷宫外轮廓
            :val2   八个宝藏点构成的列表[[迷宫坐标],半径]
        @note
        @raises            异常
    """
    # maze_bag = copy.deepcopy(_contours)
    maze_bag = []
    Dot_arr = []
    # 遍历轮廓
    for _i in range(len(_contours)):
        _area = cv2.contourArea(_contours[_i])
        if 60000 < _area < 140000:
            # cv2.drawContours(_img, _contours, _i, (0, 255, 0), 2)
            maze_bag = contours[_i]
            break
    if len(maze_bag) == 0:
        return maze_bag, 0
    # 遍历轮廓
    for _contour in _contours:
        # 计算轮廓周长
        _perimeter = cv2.arcLength(_contour, True)
        # 对轮廓进行逼近
        _approx = cv2.approxPolyDP(_contour, 0.03 * _perimeter, True)
        _area = cv2.contourArea(_contour)
        if len(_approx) > 4 and 80 < _area < 200:
            _radius = np.sqrt(_area / np.pi)
            (_x, _y), _ = cv2.minEnclosingCircle(_contour)
            _center = (int(_x), int(_y))
            if _radius != 0 and abs(1 - (_perimeter / (2 * np.pi *  _radius))) < 0.1 \
                and cv2.pointPolygonTest(maze_bag, _center, False) >= 0:
                Dot_arr.append([_center, int(_radius)])
                # cv2.circle(_img, _center, int(_radius), (0, 0, 255), -1)
    return maze_bag, Dot_arr


def Identifiy_Color(_img, lower_color, upper_color):
    """
        @brief 通过颜色识别识别出迷宫中的起点终点坐标
        @param
            :param1 原图像
            :param2 hsv颜色下边界
            :param3 hsv颜色上边界
        @return
            :val1   颜色区块的最小外接矩形轮廓 []
            :val2   颜色最小外接矩形的中心坐标[]
        @note
        @raises            异常
    """
    _box = []; _box_poi = []
    # 将图像转换为HSV颜色空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 创建掩膜以检测红色和蓝色矩形
    _mask_color = cv2.inRange(img_hsv, lower_color, upper_color)
    # 消除噪声
    _kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    _mask_color = cv2.morphologyEx(_mask_color, cv2.MORPH_OPEN, _kernel)
    # 查找轮廓
    _contours,  _ = cv2.findContours(_mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for _contour in _contours:
        # 计算轮廓周长
        _perimeter = cv2.arcLength(_contour, True)
        _area = cv2.contourArea(_contour)
        # 对轮廓进行逼近
        _approx = cv2.approxPolyDP(_contour, 0.05 * _perimeter, True)
        if len(_approx) < 8 and 300 <= _area < 1500:
            # # 最小外接矩形
            _rect = cv2.minAreaRect(_contour)
            _box_poi = [int(_rect[0][0]), int(_rect[0][1])]
            # _box = np.int32(cv2.boxPoints(_rect))
            # 修复关键行：四舍五入并转换为整数
            _box = cv2.boxPoints(_rect)
            _box = np.round(_box).astype(np.int32)

            # cv2.drawContours(_img, [_box], 0, (0, 0, 255), 2)
            break
    return _box, _box_poi


def DrawImage(_img, _thresh, maze_con, dot_arr, r_box, b_box):
    """
        @brief 将二维矩阵迷宫可视化
        @param
            :param1 原图像
            :param2 原图像二值图
            :param3 迷宫矩阵数组
            :param4 宝藏点坐标
            :param5 红色区域
            :param6 蓝色区域
        @return
            :val1 重新绘制的图像
        @note
        @raises            异常
    """
    _line_spacing = 337 / 21  # 画21条垂直线
    for i in range(21):
        for j in range(21):
            x = 79 + round(i * _line_spacing)
            y = 79 + round(j * _line_spacing)
            cv2.circle(_img, [x,y], 1, (0, 255, 255), -1)

    # 创建遮罩
    mask = np.zeros_like(_thresh)
    if len(r_box) != 0:
        cv2.drawContours(_img, [r_box],    0, (0, 0, 255), 2)
        # cv2.circle(_img, [r_box], 1, (0, 255, 255), -1)
        # cv2.drawContours(mask, [r_box], -1, 255, cv2.FILLED)
    if len(b_box) != 0:
        cv2.drawContours(_img, [b_box],    0, (255, 0, 0), 2)
        # cv2.circle(img_copy, blue_boxP, 1, (0, 255, 255), -1)
        # cv2.drawContours(mask, [b_box], -1, 255, cv2.FILLED)
    if len(maze_con) != 0:
        cv2.drawContours(_img, [maze_con], -1, (0, 255, 0), 2)
    if len(dot_arr) != 0:
        for d_con in dot_arr:
            _center, _r = d_con
            cv2.circle(_img, _center, _r, (0, 0, 255), -1)
            cv2.circle(mask, _center, _r+3, 255, -1)
    # cv2.imshow('1',mask)
    # 将遮罩与原图像相与，得到填充轮廓后的图像
    _res = cv2.bitwise_or(mask, _thresh)
    return _res


def Get_MazeMatrixArr(_thresh, _DotArr, rbox_p, bbox_p):
    """
        @brief 转换二维迷宫数组，获取对应的宝藏点坐标，获取对应起终点坐标
        @param
            :param1 源灰度图
            :param2 宝藏点像素坐标列表
            :param3 红色中心点颜色像素坐标
            :param4 蓝色中心点颜色像素坐标
        @return
            :val1   二维迷宫数组
            :val2   宝藏点坐标
            :val3   红色蓝色中心坐标
        @note
        @raises            异常
    """
    _Maze_Matrix = np.zeros((21, 21))
    # 对二值化图像进行腐蚀和膨胀操作
    _kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    _thresh1 = cv2.erode(_thresh,   _kernel, iterations=1)
    _thresh2 = cv2.dilate(_thresh1, _kernel, iterations=1)
    _thresh  = cv2.erode(_thresh2, _kernel, iterations=8)

    _line_spacing = 337 / 21  # 画21条垂直线
    for i in range(21):
        for j in range(21):
            x = 79 + round(i * _line_spacing)
            y = 79 + round(j * _line_spacing)
            _Maze_Matrix[i][j] = int(not _thresh[x, y])
           
    _Dot_Poi = [[0] * 2 for i in range(len(DotArr))]
    _sp = [0, 0]; _ep = [0, 0]
    for _i in range(len(DotArr)):
        _Dot_Poi[_i][0] = round((_DotArr[_i][0][0] - 79) / _line_spacing)
        _Dot_Poi[_i][1] = round((_DotArr[_i][0][1] - 79) / _line_spacing)

    _sp[0] = round((rbox_p[0] - 79) / _line_spacing)
    _sp[1] = round((rbox_p[1] - 79) / _line_spacing)
    _ep[0] = round((bbox_p[0] - 79) / _line_spacing)
    _ep[1] = round((bbox_p[1] - 79) / _line_spacing)
    _cp = [_sp, _ep]
    # print("起点坐标：", _sp)
    # print("终点坐标：", _ep)
    return _Maze_Matrix, _Dot_Poi, _cp, _thresh


def draw_maze(maze, pox, color_p):
 
    CELL_SIZE = 15
    CELL_PADDING = 3
    WALL_COLOR = (0, 0, 0)
    PATH_COLOR = (255, 255, 255)
    SE_COLOR = ((0, 0, 255), (255, 0, 0))
    V_COLOR = (0, 255, 255)
    # Create image
    maze_height, maze_width = maze.shape
    img_height = maze_height * CELL_SIZE + (maze_height + 1) * CELL_PADDING
    img_width = maze_width * CELL_SIZE + (maze_width + 1) * CELL_PADDING
    _img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Draw maze
    for y, row in enumerate(maze):
        for x, cell in enumerate(row):
            y_start = y * (CELL_SIZE + CELL_PADDING) + CELL_PADDING
            x_start = x * (CELL_SIZE + CELL_PADDING) + CELL_PADDING
            y_end = y_start + CELL_SIZE
            x_end = x_start + CELL_SIZE
            if cell == 1:
                # Wall
                cv2.rectangle(_img, (x_start, y_start), (x_end, y_end), WALL_COLOR, -1)
            else:
                # Path
                cv2.rectangle(_img, (x_start, y_start), (x_end, y_end), PATH_COLOR, -1)

    # Draw treasures
    for x, y in pox:
        x_start = x * (CELL_SIZE + CELL_PADDING) + CELL_PADDING
        y_start = y * (CELL_SIZE + CELL_PADDING) + CELL_PADDING
        x_end = x_start + CELL_SIZE
        y_end = y_start + CELL_SIZE
        cv2.rectangle(_img, (x_start, y_start), (x_end, y_end), V_COLOR, -1)

    # # Draw start and end points
    for i in range(len(color_p)):
        x = color_p[i][0]
        y = color_p[i][1]
        x = min(max(x, 0), maze_width  - 1)
        y = min(max(y, 0), maze_height - 1)
        x_start = x * (CELL_SIZE + CELL_PADDING) + CELL_PADDING
        y_start = y * (CELL_SIZE + CELL_PADDING) + CELL_PADDING
        x_end = x_start + CELL_SIZE
        y_end = y_start + CELL_SIZE
        cv2.rectangle(_img, (x_start, y_start), (x_end, y_end), SE_COLOR[i], -1)
    cv2.imshow('Maze', _img)


if __name__ == "__main__":
    cap = cv2.VideoCapture(2)  # 0 表示默认的摄像头
    # # 设置帧率属性
    # cap.set(cv2.CAP_PROP_FPS, 60)

    while True:
        while True:
            # 读取一帧图像#读取图像,预处理图像
            _, img = cap.read()
            # img = cv2.imread('maze_img.jpg')
            img = cv2.resize(img, (640, 480))
            img_copy = img.copy()
            contours, thresh, hierarchy, edges = PreDispose_Img(img_copy)
            Poi_Arr, Box_Arr, Loc_Num = Identify_LocalPoi(img_copy, contours, hierarchy)

            for i in range(Loc_Num):
                cx, cy = Poi_Arr[i]
                # 画出回字定位图案和定位点中心
                cv2.drawContours(img_copy, Box_Arr[i], 0, (0, 0, 255), 2)
                cv2.circle(img_copy, Poi_Arr[i], 3, (0, 255, 0), -1)

            cv2.imshow('Video', img_copy)

            if len(Poi_Arr) >= 4:
                break
            # 检测是否按下了 q 键，如果是则退出循环
            if cv2.waitKey(1) == ord('q'):
                break

        Poi_Arr = Sort_PositPoi(Poi_Arr)
        # print(Poi_Arr)
        img = Transf_Image(img, Poi_Arr)
        img_copy = img.copy()
        contours, thresh, hierarchy, edges = PreDispose_Img(img_copy)
        MazeCon, DotArr = Recognize_Dot(img_copy, contours)
        if len(MazeCon) == 0:
            continue
        lc_red = (0, 110, 100);    uc_red = (20, 255, 255)
        red_arr, red_boxP = Identifiy_Color(img_copy, lc_red, uc_red)
        lc_blue = (100, 84, 54); uc_blue = (140, 255, 255)
        blue_arr, blue_boxP = Identifiy_Color(img_copy, lc_blue, uc_blue)
        if len(red_boxP) == 0 or len(blue_boxP) == 0:
            continue
        thresh = DrawImage(img_copy, thresh, MazeCon, DotArr, red_arr, blue_arr)
        Maze_Matrix, DotPo, RBPoi, thresh = Get_MazeMatrixArr(thresh, DotArr, red_boxP, blue_boxP)
        draw_maze(Maze_Matrix, DotPo, RBPoi)

        cv2.imshow('Video', img_copy)
        cv2.imshow('Video1', thresh)
       
        while True:
            if cv2.waitKey(1) == ord('b'):
                break
            # 检测是否按下了 q 键，如果是则退出循环
        if cv2.waitKey(1) == ord('q'):
            break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
