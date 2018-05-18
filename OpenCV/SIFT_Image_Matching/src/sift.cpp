#include "sift.h"

// sift特征提取部分类成员函数cpp文件

// 建立尺度空间
sift::sift(Mat source, int num_octaves, int num_scales)
    :  source(source), num_octaves(num_octaves), num_scales(num_scales)
{
    for (int oct = 0; oct < num_octaves; ++oct)
    {
        Scale_Space.push_back(OneDMatArray(num_scales+3));   // 尺度空间层，需要n层，就建n+3层，为了保证DoG_Space最上层和最下层能计算
        DoG_Space.push_back(OneDMatArray(num_scales+2));     // DoG空间层，比上面少1层，n+2层，为了保证DoG_Space最上层和最下层能计算
        DoG_Keypts.push_back(OneDMatArray(num_scales));      // DoG关键点层
        Magnitude.push_back(OneDMatArray(num_scales));       // 幅值层，储存每个像元的梯度幅值
        Orientation.push_back(OneDMatArray(num_scales));     // 方向层，储存每个像元的梯度方向
		_Descriptors.push_back(OneDMatArray(num_scales));    // 描述子层数与图像金字塔层数相同
    }

	num_total_keypts = 0; // 初始化关键点总数
}

// 合并creat_scale_space，detect_DoG_extrema，filter_DoG_extrema，assign_orientation这4个函数，依次调用
void sift::do_sift()
{
    creat_scale_space();
    detect_DoG_extrema();
    filter_DoG_extrema();
    assign_orientation();
}

// 左右两张影像合并
Mat sift::append_images(Mat image1, Mat image2)
{
    Mat appended; int max_rows;
    if (image1.rows > image2.rows)
    {
        max_rows = image1.rows;
    }
    else
    {
        max_rows = image2.rows;
    }
    appended.create(max_rows, image1.cols + image2.cols, image1.type());
    image1.copyTo(appended(Range(0, image1.rows), Range(0, image1.cols)));
    image2.copyTo(appended(Range(0, image2.rows), Range(image1.cols, image1.cols + image2.cols)));
    return appended;
}


// 建立尺度空间
void sift::creat_scale_space()
{
    cout << "\n\n建立尺度空间..." << endl;
    Size ksize(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE);
    double sigma; Mat src_antialiased, up, down;

	// 没有必要对原图像进行升采样，会导致关键点行列号超出原图行列数，不方便匹配，而且没有升采样会快很多
	source.copyTo(Scale_Space[0][0]);

    for (int oct = 0; oct < num_octaves; oct++)
    {
        sigma = GAUSSIAN_SIGMA;  
        for (int sc = 0; sc < num_scales+2; sc++)
        {
			// 对每一层scale使用不同的sigma值，达到不同的滤波效果，方便做DoG
            sigma = sigma * pow(2.0,sc/2.0) ;      

            // 对当前octave的下一层scale进行高斯滤波
            GaussianBlur(Scale_Space[oct][sc], Scale_Space[oct][sc+1], ksize, sigma);

            // 计算DoG，用相邻两层不同程度高斯模糊的scale层进行相减
            DoG_Space[oct][sc] = Scale_Space[oct][sc] - Scale_Space[oct][sc+1]; // 这里可以解释为什么构造函数建立尺度空间时DoG_Space比Scale_Space少1层
            cout << "\tOctave : " << oct << "   Scale : " << sc << "   Scale_Space size : " << Scale_Space[oct][sc].rows << "x" << Scale_Space[oct][sc].cols << endl;
        }

		// 建立下一层octave（从金字塔看上去是上一层），下一层的行列相对于上层减半，已达到尺度不变性
        if (oct < num_octaves - 1)
        {
            pyrDown(Scale_Space[oct][0], down);
            down.copyTo(Scale_Space[oct+1][0]);
        }
    }
}


// DoG层极值点探测
// 检测当前点与周围26个点（同层8个，上下相邻层9+9个）的极值关系
// 满足条件即为关键点
void sift::detect_DoG_extrema()
{
    cout << "\n\n极值点探测..." << endl;
    Mat local_maxima, local_minima, extrema, current, top, down;

    for (int oct = 0; oct < num_octaves; oct++)
    {
        for (int sc = 0; sc < num_scales; sc++)
        {
            // 将DoG_keypts阵中元素初始化为0
            DoG_Keypts[oct][sc] = Mat::zeros(DoG_Space[oct][sc].size(), DoG_Space[oct][sc].type());
            top     = DoG_Space[oct][sc];
            current = DoG_Space[oct][sc+1]; // current从第二层开始取值
            down    = DoG_Space[oct][sc+2];
            int sx = current.rows; int sy = current.cols;

			// 寻找极大值点
			// 与同层的8个点进行比较
            local_maxima = (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(0,sx-2),Range(0,sy-2)) + 0) & // 当前点大于左上点
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(0,sx-2),Range(1,sy-1)) + 0) & // 当前点大于左点
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(0,sx-2),Range(2,sy  )) + 0) & // 当前点大于左下点
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(1,sx-1),Range(0,sy-2)) + 0) & // 当前点大于上点
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(1,sx-1),Range(2,sy  )) + 0) & // 当前点大于下点
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(2,sx  ),Range(0,sy-2)) + 0) & // 当前点大于右上点
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(2,sx  ),Range(1,sy-1)) + 0) & // 当前点大于右点
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(2,sx  ),Range(2,sy  )) + 0) ; // 当前点大于右下点

            // 与上层的9个点进行比较
            local_maxima = local_maxima & (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(0,sx-2),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(0,sx-2),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(0,sx-2),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(1,sx-1),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(1,sx-1),Range(1,sy-1))) &  // same pixel in top
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(1,sx-1),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(2,sx  ),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(2,sx  ),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(2,sx  ),Range(2,sy  )));

			// 与下层的9个点进行比较
            local_maxima = local_maxima & (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(0,sx-2),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(0,sx-2),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(0,sx-2),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(1,sx-1),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(1,sx-1),Range(1,sy-1))) &  // same pixel in down
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(1,sx-1),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(2,sx  ),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(2,sx  ),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(2,sx  ),Range(2,sy  )));

			// 寻找极小值点
			// 与同层的8个点进行比较
            local_minima = (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(0,sx-2),Range(0,sy-2))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(0,sx-2),Range(1,sy-1))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(0,sx-2),Range(2,sy  ))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(1,sx-1),Range(0,sy-2))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(1,sx-1),Range(2,sy  ))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(2,sx  ),Range(0,sy-2))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(2,sx  ),Range(1,sy-1))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(2,sx  ),Range(2,sy  ))) ;

			// 与上层的9个点进行比较
            local_minima = local_minima & (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(0,sx-2),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(0,sx-2),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(0,sx-2),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(1,sx-1),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(1,sx-1),Range(1,sy-1))) &  // same pixel in top
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(1,sx-1),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(2,sx  ),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(2,sx  ),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(2,sx  ),Range(2,sy  )));

			// 与下层的9个点进行比较
            local_minima = local_minima & (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(0,sx-2),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(0,sx-2),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(0,sx-2),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(1,sx-1),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(1,sx-1),Range(1,sy-1))) &  // same pixel in down
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(1,sx-1),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(2,sx  ),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(2,sx  ),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(2,sx  ),Range(2,sy  )));

			// 满足极大值点或者极小值点的情况都认为是极值点
            extrema = local_maxima | local_minima;

            // 极值点层比Scale_Space层少了边缘的两圈像素，将其赋值到正确的行列
            extrema.copyTo(DoG_Keypts[oct][sc](Range(1, DoG_Keypts[oct][sc].rows-1), Range(1, DoG_Keypts[oct][sc].cols-1)));
        }
    }
}

// DoG层极值点过滤（过滤低反差点和边缘点）
void sift::filter_DoG_extrema()
{
    cout << "\n\n极值点（低反差点/边缘点）滤除...\n\n";
    Mat locs; int x, y, rx, ry, fxx, fxy, fyy, deter;
    float trace, curvature;
    float curv_threshold = ((R_THRESHOLD+1)*(R_THRESHOLD+1))/R_THRESHOLD;

	// 可以设定一个大的阈值，尽量多提取出点
	//curv_threshold = 100;

    for (int oct = 0; oct < num_octaves; oct++)
    {
        for (int sc = 0; sc < num_scales; sc++)
        {
            int reject_contrast_count = 0, reject_edge_count = 0;
			cv::findNonZero(DoG_Keypts[oct][sc], locs);       // 关键点的位置（x,y坐标）

			// 判定没有提取出点的情况，若没有提取出点，则跳过对这一层的后续操作。没有必要，已经把自己造的findNonZero换成了opencv自带的函数，
			/*
			if (locs.rows == 1)
			{
				continue;
			}
			*/

            int num_keypts = locs.rows;                       // 关键点数量
            Mat_<uchar> current = DoG_Space[oct][sc+1];

            for (int k = 0; k < num_keypts; k++)
            {
				// 改过x,y，x表示行号，y表示列号
                y = locs.at<int>(k,0);
                x = locs.at<int>(k,1);

				// 放在这比放在AssignOrientation函数中合适，先约束
				if (x <= 8 || y <= 8 || x >= DoG_Keypts[oct][sc].rows - 8 || y >= DoG_Keypts[oct][sc].cols - 8)
				{
					DoG_Keypts[oct][sc].at<uchar>(x, y) = 0;
				}

                // 滤去低反差的点，通过把该点在DoG_Keypts中的强度置为0
                else if (abs(current(x,y)) < CONTRAST_THRESHOLD)
                {
                    DoG_Keypts[oct][sc].at<uchar>(x,y) = 0; 
                    reject_contrast_count++;
                }
                // 滤去边缘点
                else
                {
                    //rx = x+1;
                    //ry = y+1;   
					rx = x;
					ry = y;	// 边缘一圈像素已经被削掉了，不用担心越界

                    // 获取Hessian矩阵
                    fxx = current(rx-1,ry) + current(rx+1,ry) - 2*current(rx,ry);   // x方向的二阶差分
                    fyy = current(rx,ry-1) + current(rx,ry+1) - 2*current(rx,ry);   // y方向的二阶查分
                    fxy = current(rx-1,ry-1) + current(rx+1,ry+1) - current(rx-1,ry+1) - current(rx+1,ry-1); // x,y（45°）方向的差分
                    // 求Hessian矩阵的迹和行列式
                    trace = (float)(fxx + fyy);
                    deter = (fxx*fyy) - (fxy*fxy);
                    curvature = (float)(trace*trace/deter);

					// 满足条件，则剔除边缘点，通过把该点在DoG_Keypts中的强度置为0
                    if (deter < 0 || curvature > curv_threshold)  
                    {
                        DoG_Keypts[oct][sc].at<uchar>(x,y) = 0;
                        reject_edge_count++;
                    }

					/*
					if (num_keypts - reject_contrast_count - reject_edge_count > 1)
					{
						// 还活着的特征点存到KeyPoints中去
						KeyPoint keypoint;
						keypoint.pt.x = y;		
						keypoint.pt.y = x;
						//keypoint.id = keypt_id++;
						KeyPoints.push_back(keypoint);
					}	
					*/  // 不能再这里存入KeyPoints，因为置为0的像素也被塞进来了
                }
            }
            printf("\tOctave : %d  Scale : %d  Rejected_Keypoints : %5d  Left_Keypoints : %4d\n", oct+1, sc+1, reject_contrast_count + reject_edge_count, num_keypts - reject_contrast_count - reject_edge_count);
        }
    }
}

// 关键点方向分配（其实在这个函数中还做了些别的工作）
void sift::assign_orientation()
{
    cout << "\n关键点方向分配...\n\n";

    for (int oct = 0; oct < num_octaves; oct++)
    {
        for (int sc = 0; sc < num_scales; sc++)
        {
			// DOG在提取完关键点以后就没用了，后面的描述关键点和匹配都是在尺度空间来做的
            Mat_<uchar> current = Scale_Space[oct][sc+1];
            Magnitude[oct][sc] = Mat::zeros(current.size(), CV_32FC1);
            Orientation[oct][sc] = Mat::zeros(current.size(), CV_32FC1);
			

			// 当前层金字塔，每一像素具有梯度大小和方向
            for (int x = 1; x < current.rows-1; x++)
            {
                for (int y = 1; y < current.cols-1; y++)
                {
                    // 计算x和y方向的差分
                    double dx = current(x+1,y) - current(x-1,y); 
                    double dy = current(x,y+1) - current(x,y-1);

                    // 计算当前点的梯度幅值和方向
                    Magnitude[oct][sc].at<float>(x,y)   = sqrt(dx * dx + dy * dy);
                    Orientation[oct][sc].at<float>(x,y) = (atan2(dy, dx) == PI)? -PI : atan2(dy,dx);  // Orientation的取值是-PI到+PI

//                  printf("\tLocation[%d][%d]: Magnitude = %.3f \t Orientation = %.3f\n", x, y, Magnitude[oct][sc].at<uchar>(x,y), Orientation[oct][sc].at<uchar>(x,y));
//					printf("\tLocation[%d][%d]: Magnitude = %.3f \t Orientation = %.3f\n", x, y, Magnitude[oct][sc].at<double>(x, y), Orientation[oct][sc].at<double>(x, y));
                }
            }

			// 上面已经给当前octave和当前scale算出了所有像素的幅值和方向
			// 下面给当前octave和当前scale的关键点分配幅值和方向，利用直方图统计主方向
			
			Mat locs;
			cv::findNonZero(DoG_Keypts[oct][sc], locs);
			int x, y, num_keypts = locs.rows;
			//num_total_keypts += num_keypts; // 这么统计出来永远会多10个左右，改到下面和保存关键点操作串行

			//Descriptors在金字塔的每一层都是一个Mat，单行存储128维描述子，行数为当前层关键点数量，Descriptors本身存储所有层的描述子
			_Descriptors[oct][sc] = Mat::zeros(num_keypts, 128, CV_32FC1);

			// 直方图变量
			double* hist = new double[36]; // 统计36个方向，每10°算一个条条
			for (int i = 0; i < 36; i++)   // 初始化hist  
				*(hist + i) = 0.0;


			// 当前层octave和scale的关键点遍历
			for (int keypt = 0; keypt < num_keypts; keypt++)
			{
				// 当前关键点坐标
				y = locs.at<int>(keypt, 0); //关键点的x坐标
				x = locs.at<int>(keypt, 1); //关键点的y坐标

				/*
				// 排除在边缘的特征点，因为获取他们8*8区域的灰度时内存是炸，已经削去边缘8圈的像素了
				if (x <= 8 || y <= 8 || x >= DoG_Keypts[oct][sc].rows - 8 || y >= DoG_Keypts[oct][sc].cols - 8)
				{
					continue;
				}
				*/

				// 借道把Keypoints保存一下
				KeyPoint keypoint;
				keypoint.pt.x = y;		// !!!可能写反了????
				keypoint.pt.y = x;
				keypoint.octave = oct;    // 保存oct信息很重要，因为要依照当前层的比例缩放至原图size进行draw，sc信息不重要，不影响缩放
				KeyPoints.push_back(keypoint);
				num_total_keypts++;

				// 在关键点16*16邻域内遍历点的幅值和方向进行直方图统计
				// 设置邻域点的权，用高斯模板
				double weight, mag;
				int bar;

				for (int i = -8; i <= 8; i++)
				{
					for (int j = -8; j <= 8; j++)
					{

						double ri = (double(abs(i)) + 0.000000001) / 8.0001;  //将i,j规则化到(0,1)范围
						double rj = (double(abs(j)) + 0.000000001) / 8.0001;
						double r = sqrt(ri*ri + rj*rj);
						weight = (1 / sqrt(2 * PI)) * exp(-(r*r) / 2); // 用离散高斯函数算出来的权

						mag = Magnitude[oct][sc].at<float>(x + i, y + j);
						bar = cvRound(36 * (PI - Orientation[oct][sc].at<float>(x + i, y + j)) / (2.0 * PI));
						//printf("%d\n", bar);
						hist[bar] += mag * weight;
					}
				}
				
				//对直方图进行高斯平滑
				double prev = hist[36 - 1];
				double temp;
				double h0 = hist[0];

				for (int i = 0; i < 36; i++)
				{
					temp = hist[i];
					hist[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * (i + 1 >= 36 ? h0 : hist[i + 1]);//对方向直方图进行高斯平滑  
					prev = temp;
				}

				// 统计出直方图内最大的柱子
				double maxd = hist[0];
				int maxi = 0;
				for (int i = 1; i < 36; i++)
				{
					if (hist[i] > maxd)
					{					   						   //求取36个柱中的最大峰值  
						maxd = hist[i];							   //存最大值
						maxi = i;								   //存是第几个柱子
					}
				}

				// 计算关键点新的主方向
				double newKeyptAngle = (maxi - 18) * 10;

				// 图像旋转，旋转至主方向
				// 先试试全图旋转，TODO: 截取部分图像旋转可以提升效率
				// 得到图像大小
				int width = current.cols;
				int height = current.rows;

				//计算图像中心点
				Point2f center;
				center.x = y;
				center.y = x;

				//获得旋转变换矩阵
				double scale = 1.0;
				Mat transMat = getRotationMatrix2D(center, newKeyptAngle, scale); //newKeyptAngle参数为负表示顺时针旋转

				// 主方向旋转到正方向? 如果目前的主方向是正90°，应该把图像逆时针旋转90°? 是的
				// 应当是矫正，而不是顺应

				// 旋转后矩阵
				Mat rotated;

				//仿射变换
				warpAffine(current, rotated, transMat, Size(width, height));

				//建立当前关键点的描述子，加入整体的动态数组或者Mat
				//开16*16的像素窗口（不包括处在中心的关键点）
				//注意current已经旋转过方向了，应该对current_rotated进行处理
				//因为是绕着关键点进行旋转的，且图像尺寸没有改变，关键点所处的行列号没有改变，仍是x,y
				//还要进行高斯加权，直接用距离关键点的距离作为权
				
				//遍历16个种子点
				//对每个关键点，建立1个128维描述子，即16个种子点，每个种子点8个方向

				double descriptor[16][8];
				int seedSeq = 0; //记录遍历到第几个种子点

				for (int seedi = -6; seedi <= 6; seedi += 4)
				{
					for (int seedj = -6; seedj <= 6; seedj += 4)
					{
						//在每个种子点内遍历
						//double* seedHist = new double[8]; // 每个种子点的直方图，统计8个方向
						double* seedHist = descriptor[seedSeq]; // 当前种子点

						int seedBar;
						for (int i = 0; i < 8; i++)       // 初始化seedHist  
							*(seedHist + i) = 0.0;

						for (int i = -2; i < 2; i++)
						{
							for (int j = -2; j < 2; j++)
							{
								float dx = rotated.at<uchar>(x + seedi + i + 1, y + seedj + j) - rotated.at<uchar>(x + seedi + i - 1, y + seedj + j);
								float dy = rotated.at<uchar>(x + seedi + i, y + seedj + j + 1) - rotated.at<uchar>(x + seedi + i, y + seedj + j - 1);

								float eachMeg = sqrt(dx * dx + dy * dy);
								float eachOri = (atan2(dy, dx) == PI) ? -PI : atan2(dy, dx);
								double eachx = (double(abs(seedi + i)) + 0.000000001) / 8.0001;  //将两维距离规则化到(0,1)范围
								double eachy = (double(abs(seedj + j)) + 0.000000001) / 8.0001;
								double eachDist = sqrt(eachx*eachx + eachy*eachy);				//当前像素与关键点像素的距离
								float eachWeight = (1 / sqrt(2 * PI)) * exp(-(eachDist*eachDist) / 2); //设高斯权值
								seedBar = cvRound(8 * (PI - eachOri) / (2.0 * PI));
								seedHist[seedBar] += eachMeg * eachWeight;
							}
						}
						seedSeq++;
					}
				}

				//当前关键点周围16个种子点的直方图（128维描述子）生成完毕
				//直方图储存在descriptor[16][8]数组中
				//接下来可以把当前descriptor[16][8]存到一个大Mat(Descriptors)中，大Mat按顺序存储所有关键点的描述子
				//Descriptors在金字塔的每一层都是一个Mat，单行存储128维描述子，行数为当前层关键点数量；Descriptors本身存储所有层的描述子

				// 直方图截断，进一步抑制噪声的影响
				for (int v = 0; v < 128; v++)
				{

					double test1 = *(descriptor[0] + v);
					if (*(descriptor[0] + v) > 60)
					{
						double test2 = *(descriptor[0] + v);
						*(descriptor[0] + v) = 60;
					}
				}

				// 直方图归一化
				double cur, len_inv, len_sq = 0.00001;

				for (int v = 0; v < 128; v++)
				{
					cur = *(descriptor[0] + v);
					len_sq += cur*cur;
				}

				len_inv = 1.0 / sqrt(len_sq);

				for (int v = 0; v < 128; v++)
				{
					// 仍旧在归一化
					*(descriptor[0] + v) *= len_inv;
					// 将直方图作为描述子存入_Descriptors
					_Descriptors[oct][sc].at<float>(keypt, v) = *(descriptor[0] + v); //descriptor[0]是128维向量中第一个元素的地址
				}
				
			} // 当前层关键点遍历结束
        }
    } // 图像金字塔所有层遍历结束，描述子已经存入_Descriptors


}


// 获取关键点
vector<KeyPoint> sift::get_keypoints()
{
	return KeyPoints;
}


// 获取128维描述子
Mat sift::get_descriptors()
{
	// 初始化Descriptors阵
	Descriptors = Mat(num_total_keypts, 128, CV_32FC1);
	float* descriptorData = Descriptors.ptr<float>(0);
	int globalKeypt = 0; // 为了确定存入到Descriptors的哪一行，需要定义一个全局的关键点索引，而不是当前octave和scale的索引

	// 将_Descriptor中的描述子赋给大Mat阵Descriptors，遍历_Descriptor的每一层
	for (int oct = 0; oct < num_octaves; oct++)
	{
		for (int sc = 0; sc < num_scales; sc++)
		{
			// 还是通过访问Mat中元素进行操作
			// 在for条件中筛选掉_Descriptors中只有1个关键点的层（其实是没有关键点），不将这些假关键点加入到Descriptors中
			for (int keypt = 0; keypt < _Descriptors[oct][sc].rows && _Descriptors[oct][sc].rows != 1; keypt++) 
			{
				for (int dim = 0; dim < 128; dim++)
				{
					Descriptors.at<float>(globalKeypt, dim) = _Descriptors[oct][sc].at<float>(keypt, dim);
					float test = Descriptors.at<float>(globalKeypt, dim);
					test = 0;
				}
				globalKeypt++;
			}
		}
	}
	// 还是有必要将_Descriptors转成Descriptors的，_Descriptors中含有[没有关键点的层]，而且不如Descriptors与KeyPoints的对应关系简单
	return Descriptors;
}

// 影像显示函数
void sift::display_images(string left_or_right, int oct, int sc, int mode)
{
    stringstream ss_oct, ss_sc;
    ss_oct << oct; ss_sc << sc;

    switch (mode)
    {
        case 1:        // 显示尺度空间图像
        {
        string wSS = left_or_right + " SS Octave " + ss_oct.str() + " Scale " + ss_sc.str();
            imshow(wSS, Scale_Space[oct-1][sc-1]);
            break;
        }
        case 2:        // 显示DoG图像
        {
            string wDoG = left_or_right + " DoG_Space Octave " + ss_oct.str() + " Scale " + ss_sc.str();
            imshow(wDoG, DoG_Space[oct-1][sc-1]);
            break;
        }
        case 3:       // 显示DoG关键点
        {
            string wDoGKP = left_or_right + " DoG_Space Keypts Octave " + ss_oct.str() + " Scale " + ss_sc.str();
            imshow(wDoGKP, DoG_Keypts[oct-1][sc-1]);
            break;
        }
    }
}


// 自己造的findNonZero函数，后来在新版本的opencv(2.4.13)中找到了可直接调用的函数
/*
Mat sift::findNonZero(Mat _src)
{
	//CV_INSTRUMENT_REGION();

	//Mat src = _src.getMat();
	Mat src, _idx;
	_src.copyTo(src);


	// 删除在图像边缘（8像素以内）的非零点，将其变为0
	for (int i = 0; i < src.rows; i++)
	{
		src.at<uchar>(i, src.cols - 1) = 0;
		src.at<uchar>(i, src.cols - 2) = 0;
		src.at<uchar>(i, src.cols - 3) = 0;
		src.at<uchar>(i, src.cols - 4) = 0;
		src.at<uchar>(i, src.cols - 5) = 0;
		src.at<uchar>(i, src.cols - 6) = 0;
		src.at<uchar>(i, src.cols - 7) = 0;
		src.at<uchar>(i, src.cols - 8) = 0;

		src.at<uchar>(i, 1) = 0;
		src.at<uchar>(i, 2) = 0;
		src.at<uchar>(i, 3) = 0;
		src.at<uchar>(i, 4) = 0;
		src.at<uchar>(i, 5) = 0;
		src.at<uchar>(i, 6) = 0;
		src.at<uchar>(i, 7) = 0;
		src.at<uchar>(i, 8) = 0;
	}

	for (int j = 0; j < src.cols; j++)
	{
		src.at<uchar>(src.rows - 1, j) = 0;
		src.at<uchar>(src.rows - 2, j) = 0;
		src.at<uchar>(src.rows - 3, j) = 0;
		src.at<uchar>(src.rows - 4, j) = 0;
		src.at<uchar>(src.rows - 5, j) = 0;
		src.at<uchar>(src.rows - 6, j) = 0;
		src.at<uchar>(src.rows - 7, j) = 0;
		src.at<uchar>(src.rows - 8, j) = 0;

		src.at<uchar>(1, j) = 0;
		src.at<uchar>(2, j) = 0;
		src.at<uchar>(3, j) = 0;
		src.at<uchar>(4, j) = 0;
		src.at<uchar>(5, j) = 0;
		src.at<uchar>(6, j) = 0;
		src.at<uchar>(7, j) = 0;
		src.at<uchar>(8, j) = 0;
	}

	CV_Assert(src.type() == CV_8UC1);
	int n = countNonZero(src);
	
	
	if (n == 0)
	{
		_idx.create(1, 1, CV_8U);
		return _idx;
		//_idx.release();
		//exit(1);
	}


	/*
	_idx.create(n, 1, CV_32SC2); //两通道

	//CV_Assert(idx.isContinuous());
	Point* idx_ptr = _idx.ptr<Point>();


	for (int i = 0; i < src.rows - 1; i++)
	{
		const uchar* bin_ptr = src.ptr(i);
		for (int j = 0; j < src.cols - 1; j++)
			if (bin_ptr[j])
			{
				*idx_ptr++ = Point(j, i);
				// 用指向_idx的指针似乎改不了_idx里面存的值，试试，直接用Mat类的指针改值
			}
	}
	//*idx_ptr++ = Point(0, 0);
	return _idx;
}
*/