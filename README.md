# Identify_Maze_image

## 介绍
  该pyhton脚本是第十一届光电设计大赛迷宫小车的
  该程序是识别迷宫图像并返回该迷宫的二维数组以及藏宝点坐标和起终点坐标

## 迷宫图样

   <img src="maze_img.jpg" width="300" height="300">

## 识别图像定位点

  <img src="loc_img.jpg" width="300" height="300">

## 处理完成图像

  <img src="cope_img.jpg" width="300" height="300">

## 遮盖宝藏点后二值化闭运算后的图像

  <img src="thresh_img.jpg" width="300" height="300">

## 获得迷宫数据并可视化迷宫

  <img src="re_maze_img.jpg" width="300" height="300">

## 日志:

```
 Version : 0.5
 Time    : 2025.03.29.17:33
 Note    : 修复部分bug，提升识别准确性
           完成文档修改
           
```

```
 Version : 0.4
 Time    : 2023.04.26.1:52
 Note    : 对drawMaze函数进行修改.除了drawImage以外所有的函数都不应
           该在函数体里面直接更改img,drawImag有点问题。
           修改了README.md,重命名了图像名称。

```

```
 Version : 0.3
 Time    : 2023.04.25.23:55
 Note    :本次将迷宫转化为二维数组的代码需要封装为函数。
          完善的README.md文件

```

```
 Version : 0.2
 Time    : 2023.04.25.16.53
 Note    :本次将函数注释完成，部分函数有些问题不应该在函数内更改原图像，同时主函数中将迷宫转化为
         二维数组的代码需要封装为函数，有时候程序会莫名其妙崩溃，有可能是应为列表为空后使用cv函数
         调用发生崩溃，后期将所有函数放到一个包内。
```
