//
//  houghCircle.hpp
//  coin
//
//  Created by Yifan Jin on 2018/11/27.
//  Copyright © 2018 Yifan Jin. All rights reserved.
//

#ifndef hough_hpp
#define hough_hpp

#include <stdio.h>

#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <set>

using namespace cv;
using namespace std;

typedef vector<float> RadiusVoting;
typedef vector<vector<RadiusVoting>> HoughCircleVoting;
typedef pair<int, int> Position;

typedef pair<Position, int> Circle;
typedef pair<float, float> Line;

class Hough{
    string fileName;
    // 算梯度
    Mat gradientMag;
    Mat gradientDir;
    
    // 梯度矩阵的长宽
    int rows;
    int cols;
    
    // for circle detection
    // 记录 Hough 空间的投票结果
    Mat houghCircleAcc;
    // 记录识别出的圆，一个圆由 x，y，r 表示
    vector<Circle> circles;
    // 记录圆的向量对应 index 的投票结果
    vector<float> circleVotes;
    // 检测的参数，决定投票的时候，半径纬度的起始和结束
    int minR;
    int maxR;
    // 用于非极大抑制，识别出的待定圆心可能会挤在一起
    // 所以在一定距离内，寻找 Hough 投票结果最大的圆
    // 把其他不是极大的点都忽略掉
    int distance;
    // 对于每一个坐标点，记录下从 minR 到 maxR 的半径条件下，获得的投票
    // 把半径纬度的值加起来，就是上面的 houghCircleAcc
    HoughCircleVoting houghCircleVoting;
    
    // for line detection
    // 记录 Hough 空间投票结果
    Mat houghLineAcc;
    // 记录识别出的直线，一条直线由 rho，theta 极坐标表示，其中 theta 是弧度制，从 0 到 pi
    vector<Line> lines;
    // 对于每两条直线，计算它们的交点
    vector<Point> crossPoints;
    // 暂时没用到
    Mat cannyEdge;
    
    // 记录下最终识别的结果，包括了判断 VJ 圆、Hough 圆、以及和 Hough 直线交点的结果
    vector<Circle> combined;
    
    // 用于查看数组
    void printMat(Mat A);
    // 用来初始化生成梯度数组
    void sobel();
    
    // 计算直线交点
    Point getCrossPoint(Line line1, Line line2);
    // 计算直线相似度
    bool lineSimilar(Line line1, Line line2);
    
public:
    // 初始化 Hough，包括用 sobel 生成梯度
    Hough(string file);
    // 用于查看
    void printMag();
    void printDir();
    
    // 初始化圆检测，包括初始化投票矩阵
    void houghCircleInit(int min, int max, int dist);
    // 扫描梯度图，每一点用 Hough 圆映射，朝着正反梯度方向移动，只计算自身色值超过 th 的
    void houghCircleVote(float th);
    // 从 Hough 空间识别出大于 outThre 的所有圆，然后进行非极大抑制
    // 对于每一个圆心，以圆心为中心，distance 为半径，扫描周围 distance * distance 的像素点
    // 只保留其中的最大值，设为 255
    void houghCircleDetect(float outThre);
    // 画出 Hough 识别的所有圆
    void houghCircleDraw(string outFileName);
    
    // 初始化直线检测
    void houghLineInit();
    // 扫描梯度图，对于每一色值大于 th 的点，theta 从 0 到 360 度循环，使用 Hough 直线映射计算出 rho
    // 把所有点映射到 rho-theta 空间
    void houghLineVote(float th);
    // 把 rho-theta 空间中色值大于 outThre 的点保留，映射回 x-y 空间 push 到直线向量中
    // 然后使用二重循环，把直线向量中的直线两两比对，合并过于重合的线，也就是旋转角度 theta 差值不多，同时离原点距离 rho 差值也不高的线
    void houghLineDetect(float outThre);
    // 画线的同时，计算出交点，push 到交点矩阵中
    // 再画出交点
    void houghLineDraw(string outFileName);
    
    // 将 VJ 圆、Hough 圆和 Hough 交点合并查找
    // 1、首先对于每一个 Hough 圆圆心，扫描所有的 VJ 圆圆心，若是某一个 VJ 圆相交，则保留当前的 Hough 圆
    // 2、对于每一个 Hough 圆圆心，扫描所有的 Hough 交点，若是出现在当前圆内的 Hough 交点超过阈值，则保留当前的 Hough 圆
    // 3、对于每一个 VJ 圆圆心，扫描所有的 Hough 交点，若是出现在当前圆内的 Hough 交点超过阈值，则保留当前的 VJ 圆
    // 然后绘制所有识别出来的以上三类圆心
    void houghCombine(string outFileName, vector<Circle> VJcircles);
};


#endif /* hough_hpp */
