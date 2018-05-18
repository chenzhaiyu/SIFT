#include "sift.h"
#include "match.h"

#define NUM_OCTAVES 4
#define NUM_SCALES 5


int main(int argc, char* argv[])
{

    // 读入图像
    Mat left_image = imread("data/image1.jpg", CV_8UC1);
    Mat right_image = imread("data/image2.jpg", CV_8UC1);

	// 实例化左右影像sift对象
    sift left_detector(left_image, NUM_OCTAVES, NUM_SCALES);
	sift right_detector(right_image, NUM_OCTAVES, NUM_SCALES);

	// sift操作（依次调用4个函数）
    left_detector.do_sift();
	right_detector.do_sift();

	Mat left_descriptors;				// 左影像描述子
	Mat right_descriptors;				// 右影像描述子
	vector<KeyPoint> left_keypoints;	// 左影像关键点
	vector<KeyPoint> right_keypoints;	// 右影像关键点

	// 获取关键点
	left_keypoints = left_detector.get_keypoints();
	right_keypoints = right_detector.get_keypoints();

	// 获取描述子
	left_descriptors = left_detector.get_descriptors();
	right_descriptors = right_detector.get_descriptors();

	// 显示关键点在某一层octave和scale的位置
	left_detector.display_images("Left", 2, 4, 3); 
	right_detector.display_images("Right", 2, 4, 3);
	
	/*
	for (int i = 0; i < 100; i++)
		cout << right_descriptors.at<float>(0, i) << endl;
	*/

	// 同名点匹配
	Matcher matcher;
	matcher.match(left_descriptors, right_descriptors);
	matcher.draw(left_image, right_image, left_keypoints, right_keypoints);

    return 0;
}
