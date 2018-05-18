#include "match.h"

// siftƥ�䲿��cpp�ļ�

// ͬ����ƥ�亯��
void Matcher::match(Mat& left_descriptors, Mat& right_descriptors)
{
	BruteForceMatcher<L2<float>> BFmatcher;
	// ���ڽ����ڽ�֮�ȣ�ratioԽС����matchԽunique
	const float ratio = 0.3;
	// ֱ��ʹ����opencv�е�knnMatch����ƥ�䣬���Լ�д��128ά����ŷʽ����ƥ��Ч�ʸߺܶ࣬�ҷ���������ڽ����ڽ���ֵԼ��
	BFmatcher.knnMatch(left_descriptors, right_descriptors, NDDRMatches, 2);

	for (int n = 0; n < NDDRMatches.size(); n++)
	{
		DMatch& bestmatch = NDDRMatches[n][0];
		DMatch& bettermatch =NDDRMatches[n][1];
		// ɸѡ�����������ĵ�
		if (bestmatch.distance < ratio*bettermatch.distance)
		{
			//�����������ĵ㱣����matches������
			finalMatches.push_back(bestmatch);
		}
	}
	cout << "ƥ����: " << finalMatches.size() << endl;

}


// ͬ������ƺ���
void Matcher::draw(Mat& leftImage, Mat& rightImage, vector<KeyPoint> leftkeypoints, vector<KeyPoint> rightkeypoints)
{
	Mat img_matches;
	drawMatches(leftImage, leftkeypoints, rightImage, rightkeypoints,
		finalMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("result", img_matches);
	waitKey(0);
	
}