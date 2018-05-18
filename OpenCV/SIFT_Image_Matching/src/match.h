#pragma once

// siftƥ�䲿��ͷ�ļ�

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

	void match(Mat& left_descriptors, Mat& right_descriptors);// ͬ����ƥ�亯��
	void draw(Mat& leftImage, Mat& rightImage, vector<KeyPoint> leftkeypoints, vector<KeyPoint> rightkeypoints); // ͬ������ƺ���
	
private:
	vector<DMatch> finalMatches;        // �����������ĵ�
	vector<vector<DMatch>> NDDRMatches; // ������ڵ�ʹν��ڵ�

};