#pragma once

// sift特征提取部分头文件

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <math.h>
#include <opencv2/features2d/features2d.hpp>

#define GAUSSIAN_KERNEL_SIZE    3	// 高斯核尺寸，用于金字塔不同Scale间高斯滤波 
#define CONTRAST_THRESHOLD      5	// 反差系数阈值，用于滤去低反差的特征点
#define GAUSSIAN_SIGMA          0.5 // 高斯模板的标准差（SIGMA越小提取的特征点越多）
#define PI                      3.1415926535897932384626433832795
#define E						2.71828
#define R_THRESHOLD             5.0 // 曲率阈值，用于滤去边缘效应强的特征点（R越大点越多）

using namespace cv;
using namespace std;

// 定义两个用于存储Mat的容器
typedef std::vector<Mat> OneDMatArray;			// 一维Mat容器
typedef std::vector<OneDMatArray> TwoDMatArray;	// 二维Mat容器

// 原先定义的MyKeyPoint数据结构，后来直接使用opencv中的KeyPoint数据结构，两者的成员差不多
/*
struct MyKeyPoint
{
	int oct;
	int sc;
	int x; // x表示行号
	int y; // y表示列号
	//int id; // 绝对序号，从0开始，没有必要，直接用vector<KeyPoint>KeyPoints的数组索引确定
};
*/

// sift类
class sift
{
private:
    Mat source;					// 输入的图像
    int num_octaves;			// octave的层数（通过降采样）
    int num_scales;				// scale的层数（通过高斯模糊）
	int num_total_keypts;		// 特征点的总数
	Mat Descriptors;			// 128维描述子
	vector<KeyPoint> KeyPoints; // 用于存储关键点的容器，直接存的opencv内置KeyPoint数据类型

    // 定义二维的Mat们
    TwoDMatArray Scale_Space, DoG_Space, DoG_Keypts, Magnitude, Orientation, _Descriptors; //_Descriptors在金字塔的每层都是一个Mat，相对于Descriptor有必要建，为了和KeyPoints能够按序号对应起来
    void creat_scale_space();	// 建立尺度空间
    void detect_DoG_extrema();  // DoG层极值点探测
    void filter_DoG_extrema();  // DoG层极值点过滤（过滤低反差点和边缘点）
    void assign_orientation();  // 关键点方向分配（其实在这个函数中还做了些别的工作）

public:
    sift(Mat source, int num_octaves, int num_scales);						// 构造函数
    void do_sift();															// 合并creat_scale_space，detect_DoG_extrema，filter_DoG_extrema，assign_orientation这4个函数，依次调用
    Mat append_images(Mat image1, Mat image2);								// 左右两张影像合并
    void display_images(string left_or_right, int oct, int sc, int mode);   // 影像显示函数
	Mat get_descriptors();												    // 获取128维描述子
	vector<KeyPoint>get_keypoints();										// 获取关键点

};
