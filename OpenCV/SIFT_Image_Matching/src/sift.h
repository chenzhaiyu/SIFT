#pragma once

// sift������ȡ����ͷ�ļ�

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

#define GAUSSIAN_KERNEL_SIZE    3	// ��˹�˳ߴ磬���ڽ�������ͬScale���˹�˲� 
#define CONTRAST_THRESHOLD      5	// ����ϵ����ֵ��������ȥ�ͷ����������
#define GAUSSIAN_SIGMA          0.5 // ��˹ģ��ı�׼�SIGMAԽС��ȡ��������Խ�ࣩ
#define PI                      3.1415926535897932384626433832795
#define E						2.71828
#define R_THRESHOLD             5.0 // ������ֵ��������ȥ��ԵЧӦǿ�������㣨RԽ���Խ�ࣩ

using namespace cv;
using namespace std;

// �����������ڴ洢Mat������
typedef std::vector<Mat> OneDMatArray;			// һάMat����
typedef std::vector<OneDMatArray> TwoDMatArray;	// ��άMat����

// ԭ�ȶ����MyKeyPoint���ݽṹ������ֱ��ʹ��opencv�е�KeyPoint���ݽṹ�����ߵĳ�Ա���
/*
struct MyKeyPoint
{
	int oct;
	int sc;
	int x; // x��ʾ�к�
	int y; // y��ʾ�к�
	//int id; // ������ţ���0��ʼ��û�б�Ҫ��ֱ����vector<KeyPoint>KeyPoints����������ȷ��
};
*/

// sift��
class sift
{
private:
    Mat source;					// �����ͼ��
    int num_octaves;			// octave�Ĳ�����ͨ����������
    int num_scales;				// scale�Ĳ�����ͨ����˹ģ����
	int num_total_keypts;		// �����������
	Mat Descriptors;			// 128ά������
	vector<KeyPoint> KeyPoints; // ���ڴ洢�ؼ����������ֱ�Ӵ��opencv����KeyPoint��������

    // �����ά��Mat��
    TwoDMatArray Scale_Space, DoG_Space, DoG_Keypts, Magnitude, Orientation, _Descriptors; //_Descriptors�ڽ�������ÿ�㶼��һ��Mat�������Descriptor�б�Ҫ����Ϊ�˺�KeyPoints�ܹ�����Ŷ�Ӧ����
    void creat_scale_space();	// �����߶ȿռ�
    void detect_DoG_extrema();  // DoG�㼫ֵ��̽��
    void filter_DoG_extrema();  // DoG�㼫ֵ����ˣ����˵ͷ����ͱ�Ե�㣩
    void assign_orientation();  // �ؼ��㷽����䣨��ʵ����������л�����Щ��Ĺ�����

public:
    sift(Mat source, int num_octaves, int num_scales);						// ���캯��
    void do_sift();															// �ϲ�creat_scale_space��detect_DoG_extrema��filter_DoG_extrema��assign_orientation��4�����������ε���
    Mat append_images(Mat image1, Mat image2);								// ��������Ӱ��ϲ�
    void display_images(string left_or_right, int oct, int sc, int mode);   // Ӱ����ʾ����
	Mat get_descriptors();												    // ��ȡ128ά������
	vector<KeyPoint>get_keypoints();										// ��ȡ�ؼ���

};
