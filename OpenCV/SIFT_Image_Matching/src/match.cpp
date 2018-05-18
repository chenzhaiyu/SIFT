#include "match.h"

// sift匹配部分cpp文件

// 同名点匹配函数
void Matcher::match(Mat& left_descriptors, Mat& right_descriptors)
{
	BruteForceMatcher<L2<float>> BFmatcher;
	// 最邻近次邻近之比，ratio越小，该match越unique
	const float ratio = 0.3;
	// 直接使用了opencv中的knnMatch进行匹配，比自己写的128维向量欧式距离匹配效率高很多，且方便进行最邻近次邻近比值约束
	BFmatcher.knnMatch(left_descriptors, right_descriptors, NDDRMatches, 2);

	for (int n = 0; n < NDDRMatches.size(); n++)
	{
		DMatch& bestmatch = NDDRMatches[n][0];
		DMatch& bettermatch =NDDRMatches[n][1];
		// 筛选出符合条件的点
		if (bestmatch.distance < ratio*bettermatch.distance)
		{
			//将符合条件的点保存在matches容器中
			finalMatches.push_back(bestmatch);
		}
	}
	cout << "匹配数: " << finalMatches.size() << endl;

}


// 同名点绘制函数
void Matcher::draw(Mat& leftImage, Mat& rightImage, vector<KeyPoint> leftkeypoints, vector<KeyPoint> rightkeypoints)
{
	Mat img_matches;
	drawMatches(leftImage, leftkeypoints, rightImage, rightkeypoints,
		finalMatches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("result", img_matches);
	waitKey(0);
	
}