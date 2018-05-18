#pragma once

// sift匹配部分头文件

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string>
#include <math.h>
#include "sift.h"

using namespace cv;
using namespace std;

class Matcher
{

public:

	void match(Mat& left_descriptors, Mat& right_descriptors);// 同名点匹配函数
	void draw(Mat& leftImage, Mat& rightImage, vector<KeyPoint> leftkeypoints, vector<KeyPoint> rightkeypoints); // 同名点绘制函数
	
private:
	vector<DMatch> finalMatches;        // 存满足条件的点
	vector<vector<DMatch>> NDDRMatches; // 存最近邻点和次近邻点

};