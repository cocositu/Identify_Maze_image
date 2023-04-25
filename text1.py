import cv2
import numpy as np
import copy

# Version : 0.2
# Time    : 2023.04.25.16.53
# Note    :本次将函数注释完成，部分函数有些问题不应该在函数内更改原图像，同时主函数中将迷宫转化为
#         二维数组的代码需要封装为函数，后期将所有函数放到一个包内。


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
    #cv2.imshow('th', thresh)
    return _contours, _hierarchy, _thresh, _edges


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
    LocPo_Num = 0; PosPoi = []
    for _i in range(len(_contours)):
        # d1一级嵌套， d2二级嵌套
        d1 = _hierarchy[0][_i][2]
        d2 = _hierarchy[0][d1][2]

        _perimeter = cv2.arcLength(_contours[_i], True)
        _approx = cv2.approxPolyDP(_contours[_i], 0.05 * _perimeter, True)
        _area = cv2.contourArea(_approx)

        if 400 < _area < 1500 and d1 != -1 and d2!= -1:
            LocPo_Num += 1
            # 获取回字定位图案的最小外接矩形
            _rect = cv2.minAreaRect(_contours[_i])
            _box = np.int0(cv2.boxPoints(_rect))

            # 画出回字定位图案和定位点中心
            cx, cy =  Cmp_CentPos(_contours[_i])
            cv2.drawContours(_img, [_box], 0, (0, 0, 255), 2)
            cv2.circle(_img, (cx, cy), 3, (0, 255, 0), -1)
            # 定位点坐标
            PosPoi.append([cx, cy])

    return _img, PosPoi


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
    dst_points = np.float32([[30, 30], [30, 446], [446, 446], [446, 30]])
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    # 进行透视变换，边缘使用白色填充
    rec_img = cv2.warpPerspective(_img, M, (476, 476), borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])
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
    maze_bag = copy.deepcopy(_contours)
    Dot_arr = []
    # 遍历轮廓
    for _i in range(len(_contours)):
        _area = cv2.contourArea(_contours[_i])
        if 60000 < _area < 140000:
            # cv2.drawContours(_img, _contours, _i, (0, 255, 0), 2)
            maze_bag = contours[_i]
            break
    if len(maze_bag) == 0: return
    # 遍历轮廓
    for _contour in _contours:
        # 计算轮廓周长
        _perimeter = cv2.arcLength(_contour, True)
        # 对轮廓进行逼近
        _approx = cv2.approxPolyDP(_contour, 0.02 * _perimeter, True)
        _area = cv2.contourArea(_contour)
        if len(_approx) > 4 and 80 < _area < 200:
            _radius = np.sqrt(_area / np.pi)
            (_x, _y), _ = cv2.minEnclosingCircle(_contour)
            _center = (int(_x), int(_y))
            if  _radius != 0 and abs(1 - (_perimeter / (2 * np.pi *  _radius))) < 0.1 \
                           and cv2.pointPolygonTest(maze_bag, _center,False) >= 0:
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
    for _contour in  _contours:
        # 计算轮廓周长
        _perimeter = cv2.arcLength(_contour, True)
        _area = cv2.contourArea(_contour)
        # 对轮廓进行逼近
        _approx = cv2.approxPolyDP(_contour, 0.05 * _perimeter, True)
        if len(_approx) < 8 and 300 <= _area < 1500:
            # # 最小外接矩形
            _rect = cv2.minAreaRect(_contour)
            _box_poi = [int(_rect[0][0]), int(_rect[0][1])]
            _box = np.int0(cv2.boxPoints(_rect))
            # cv2.drawContours(_img, [_box], 0, (0, 0, 255), 2)
            break
    return _box, _box_poi


"""
    @brief 将迷宫二维矩阵，起终点，宝藏点可视化
    @param
        :param1 原图像 
        :param2 图像的所有轮廓
    @return
        :val1 
        :val2
        :val3 
    @note   
    @raises            异常
"""
def DrawImage(_img, _thresh, maze_con, dot_arr, r_box, b_box):
    # 创建遮罩
    mask = np.zeros_like(_thresh)
    if len(r_box)!=0:
        cv2.drawContours(_img, [r_box],    0, (0, 0, 255), 2)
        # cv2.drawContours(mask, [r_box], -1, 255, cv2.FILLED)
    if len(b_box)!=0:
        cv2.drawContours(_img, [b_box],    0, (255, 0, 0), 2)
        # cv2.drawContours(mask, [b_box], -1, 255, cv2.FILLED)
    if len(maze_con) != 0:
        cv2.drawContours(_img, [maze_con], -1, (0, 255, 0), 2)
    if len(dot_arr)!=0:
        for d_con in dot_arr:
            _center, _r = d_con
            cv2.circle(_img, _center, _r, (0, 0, 255), -1)
            cv2.circle(mask, _center, _r+3 , 255, -1)
    # cv2.imshow('1',mask)
    # 将遮罩与原图像相与，得到填充轮廓后的图像
    _res = cv2.bitwise_or(mask, _thresh)
    return _res


def draw_maze(maze, pox, color_p):
    """
        @brief 找出迷宫外轮廓，八个宝藏的轮廓和坐标
        @param
            :param1 迷宫二维数组
            :param2 宝藏点坐标列表
            :param3 起终点颜色坐标
        @return
        @note
        @raises            异常
    """
    # 定义颜色
    wall_color = (0, 0, 0)
    path_color = (255, 255, 255)
    start_color = (0, 255, 0)
    end_color = (0, 0, 255)
    v_color = (0,255,255)
    # 定义每个位置的大小和间隙
    cell_size = 15
    cell_padding = 3

    # 创建图像
    maze_height, maze_width = maze.shape
    img = np.zeros((maze_height * cell_size + (maze_height + 1) * cell_padding,
                    maze_width * cell_size + (maze_width + 1) * cell_padding, 3), dtype=np.uint8)

    # 遍历迷宫二维矩阵，根据颜色绘制图像
    for y in range(maze_height):
        for x in range(maze_width):
            if maze[y, x] == 1:
                # 墙
                img[y * (cell_size + cell_padding) + cell_padding:(y + 1) * (cell_size + cell_padding),
                x * (cell_size + cell_padding) + cell_padding:(x + 1) * (cell_size + cell_padding)] = wall_color
            else:
                # 路径
                img[
                y * (cell_size + cell_padding) + cell_padding:y * (cell_size + cell_padding) + cell_padding + cell_size,
                x * (cell_size + cell_padding) + cell_padding:x * (
                            cell_size + cell_padding) + cell_padding + cell_size] = path_color

    for p in pox:
        img[p[1] * (cell_size + cell_padding) + cell_padding:(p[1] + 1) * (cell_size + cell_padding),
        p[0] * (cell_size + cell_padding) + cell_padding:(p[0] + 1) * (cell_size + cell_padding)] = v_color

    for p in color_p:
        if p[0] == -1:  p[0]+=1
        if p[1] == -1:  p[1]+= 1
        if p[0] == 21:  p[0]-=1
        if p[1] == 21:  p[1]-=1
        img[p[1] * (cell_size + cell_padding) + cell_padding:(p[1] + 1) * (cell_size + cell_padding),
        p[0] * (cell_size + cell_padding) + cell_padding:(p[0] + 1) * (cell_size + cell_padding)] = end_color
    # 显示图像
    cv2.imshow('Maze', img)


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0 表示默认的摄像头
    # # 设置帧率属性
    # cap.set(cv2.CAP_PROP_FPS, 60)

    while True:
        while True:
            # 读取一帧图像#读取图像,预处理图像
            _, img = cap.read()
            # img = cv2.imread('image1.jpg')
            img = cv2.resize(img, (640, 480))
            img_copy = img.copy()
            contours, hierarchy, thresh, edges = PreDispose_Img(img_copy)
            img_copy, Poi_Arr = Identify_LocalPoi(img_copy, contours, hierarchy)

            # # 读取帧率属性
            # fps = cap.get(cv2.CAP_PROP_FPS)
            # print(fps)
            # 显示图像
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
        contours, hierarchy, thresh, edges = PreDispose_Img(img_copy)
        MazeCon, DotArr = Recognize_Dot(img_copy, contours)
        if len(MazeCon) == 0: continue
        lc_red = (0, 110, 100);    uc_red = (20, 255, 255)
        red_arr, red_boxP = Identifiy_Color(img_copy, lc_red, uc_red)
        lc_blue = (100,84, 54); uc_blue = (140, 255, 255)
        blue_arr, blue_boxP = Identifiy_Color(img_copy, lc_blue, uc_blue)

        thresh = DrawImage(img_copy, thresh, MazeCon, DotArr, red_arr, blue_arr)

        # 对二值化图像进行腐蚀和膨胀操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.erode(thresh, kernel, iterations = 1)
        thresh = cv2.dilate(thresh, kernel, iterations= 1)
        thresh = cv2.erode(thresh, kernel, iterations = 8)


        if len(red_boxP)==0 or len(blue_boxP)==0: continue
        line_spacing = 416 // 26
        # 画27条垂直线
        Maze_Matrix = np.zeros((21, 21))
        for i in range(0, 21):
            x = 84 + i * line_spacing - 5
            for j in range (0, 21):
                y = 84 + j * line_spacing - 5
                cv2.circle(img_copy, [x, y], 1, (0,255,255), -1)
                pixel_value = thresh[x, y]
                if pixel_value == 0:
                    Maze_Matrix[i][j] = 1
                else:
                    Maze_Matrix[i][j] = 0

        po = [[0] * 2 for i in range(len(DotArr))]
        for i in range(len(DotArr)):
            po[i][0] = round((DotArr[i][0][0] - 81) / 16)
            po[i][1] = round((DotArr[i][0][1] - 81) / 16)
            print(po[i],end=' ')
        print("\n")
        for i in range(0, 21):
            for j in range(0, 21):
                print("%d" %Maze_Matrix[i][j],end = ' ')
            print("")
        print("\n")
        start_P = [0,0]; end_P = [0,0]
        cv2.circle(img_copy, red_boxP, 1, (0, 255, 255), -1)
        cv2.circle(img_copy, blue_boxP, 1, (0, 255, 255), -1)
        start_P[0] = round((red_boxP[0]  - 80) / 16)
        start_P[1] = round((red_boxP[1]  - 80) / 16)
        end_P[0]   = round((blue_boxP[0] - 80) / 16)
        end_P[1]   = round((blue_boxP[1] - 80) / 16)
        cp = [start_P , end_P]
        print(cp,"\n")
        draw_maze(Maze_Matrix, po, cp)

        cv2.imshow('Video', img_copy)
        cv2.imshow('thresh', thresh)

        while True:
            if cv2.waitKey(1) == ord('b'):
                break
            # 检测是否按下了 q 键，如果是则退出循环
        if cv2.waitKey(1) == ord('q'):
                break

    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

## 后面可以创建函数使用
# # 创建空白的RGB图像
# rgb_img = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
# # 将二值图像赋值给RGB图像的三个通道
# rgb_img[:, :, 0] = thresh
# rgb_img[:, :, 1] = thresh
# rgb_img[:, :, 2] = thresh