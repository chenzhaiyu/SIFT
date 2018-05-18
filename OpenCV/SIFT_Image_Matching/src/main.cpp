#include "sift.h"
#include "match.h"

#define NUM_OCTAVES 4
#define NUM_SCALES 5


int main(int argc, char* argv[])
{

    // ����ͼ��
    Mat left_image = imread("data/image1.jpg", CV_8UC1);
    Mat right_image = imread("data/image2.jpg", CV_8UC1);

	// ʵ��������Ӱ��sift����
    sift left_detector(left_image, NUM_OCTAVES, NUM_SCALES);
	sift right_detector(right_image, NUM_OCTAVES, NUM_SCALES);

	// sift���������ε���4��������
    left_detector.do_sift();
	right_detector.do_sift();

	Mat left_descriptors;				// ��Ӱ��������
	Mat right_descriptors;				// ��Ӱ��������
	vector<KeyPoint> left_keypoints;	// ��Ӱ��ؼ���
	vector<KeyPoint> right_keypoints;	// ��Ӱ��ؼ���

	// ��ȡ�ؼ���
	left_keypoints = left_detector.get_keypoints();
	right_keypoints = right_detector.get_keypoints();

	// ��ȡ������
	left_descriptors = left_detector.get_descriptors();
	right_descriptors = right_detector.get_descriptors();

	// ��ʾ�ؼ�����ĳһ��octave��scale��λ��
	left_detector.display_images("Left", 2, 4, 3); 
	right_detector.display_images("Right", 2, 4, 3);
	
	/*
	for (int i = 0; i < 100; i++)
		cout << right_descriptors.at<float>(0, i) << endl;
	*/

	// ͬ����ƥ��
	Matcher matcher;
	matcher.match(left_descriptors, right_descriptors);
	matcher.draw(left_image, right_image, left_keypoints, right_keypoints);

    return 0;
}
