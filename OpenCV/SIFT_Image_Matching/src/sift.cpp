#include "sift.h"

// sift������ȡ�������Ա����cpp�ļ�

// �����߶ȿռ�
sift::sift(Mat source, int num_octaves, int num_scales)
    :  source(source), num_octaves(num_octaves), num_scales(num_scales)
{
    for (int oct = 0; oct < num_octaves; ++oct)
    {
        Scale_Space.push_back(OneDMatArray(num_scales+3));   // �߶ȿռ�㣬��Ҫn�㣬�ͽ�n+3�㣬Ϊ�˱�֤DoG_Space���ϲ�����²��ܼ���
        DoG_Space.push_back(OneDMatArray(num_scales+2));     // DoG�ռ�㣬��������1�㣬n+2�㣬Ϊ�˱�֤DoG_Space���ϲ�����²��ܼ���
        DoG_Keypts.push_back(OneDMatArray(num_scales));      // DoG�ؼ����
        Magnitude.push_back(OneDMatArray(num_scales));       // ��ֵ�㣬����ÿ����Ԫ���ݶȷ�ֵ
        Orientation.push_back(OneDMatArray(num_scales));     // ����㣬����ÿ����Ԫ���ݶȷ���
		_Descriptors.push_back(OneDMatArray(num_scales));    // �����Ӳ�����ͼ�������������ͬ
    }

	num_total_keypts = 0; // ��ʼ���ؼ�������
}

// �ϲ�creat_scale_space��detect_DoG_extrema��filter_DoG_extrema��assign_orientation��4�����������ε���
void sift::do_sift()
{
    creat_scale_space();
    detect_DoG_extrema();
    filter_DoG_extrema();
    assign_orientation();
}

// ��������Ӱ��ϲ�
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


// �����߶ȿռ�
void sift::creat_scale_space()
{
    cout << "\n\n�����߶ȿռ�..." << endl;
    Size ksize(GAUSSIAN_KERNEL_SIZE, GAUSSIAN_KERNEL_SIZE);
    double sigma; Mat src_antialiased, up, down;

	// û�б�Ҫ��ԭͼ��������������ᵼ�¹ؼ������кų���ԭͼ��������������ƥ�䣬����û�����������ܶ�
	source.copyTo(Scale_Space[0][0]);

    for (int oct = 0; oct < num_octaves; oct++)
    {
        sigma = GAUSSIAN_SIGMA;  
        for (int sc = 0; sc < num_scales+2; sc++)
        {
			// ��ÿһ��scaleʹ�ò�ͬ��sigmaֵ���ﵽ��ͬ���˲�Ч����������DoG
            sigma = sigma * pow(2.0,sc/2.0) ;      

            // �Ե�ǰoctave����һ��scale���и�˹�˲�
            GaussianBlur(Scale_Space[oct][sc], Scale_Space[oct][sc+1], ksize, sigma);

            // ����DoG�����������㲻ͬ�̶ȸ�˹ģ����scale��������
            DoG_Space[oct][sc] = Scale_Space[oct][sc] - Scale_Space[oct][sc+1]; // ������Խ���Ϊʲô���캯�������߶ȿռ�ʱDoG_Space��Scale_Space��1��
            cout << "\tOctave : " << oct << "   Scale : " << sc << "   Scale_Space size : " << Scale_Space[oct][sc].rows << "x" << Scale_Space[oct][sc].cols << endl;
        }

		// ������һ��octave���ӽ���������ȥ����һ�㣩����һ�������������ϲ���룬�Ѵﵽ�߶Ȳ�����
        if (oct < num_octaves - 1)
        {
            pyrDown(Scale_Space[oct][0], down);
            down.copyTo(Scale_Space[oct+1][0]);
        }
    }
}


// DoG�㼫ֵ��̽��
// ��⵱ǰ������Χ26���㣨ͬ��8�����������ڲ�9+9�����ļ�ֵ��ϵ
// ����������Ϊ�ؼ���
void sift::detect_DoG_extrema()
{
    cout << "\n\n��ֵ��̽��..." << endl;
    Mat local_maxima, local_minima, extrema, current, top, down;

    for (int oct = 0; oct < num_octaves; oct++)
    {
        for (int sc = 0; sc < num_scales; sc++)
        {
            // ��DoG_keypts����Ԫ�س�ʼ��Ϊ0
            DoG_Keypts[oct][sc] = Mat::zeros(DoG_Space[oct][sc].size(), DoG_Space[oct][sc].type());
            top     = DoG_Space[oct][sc];
            current = DoG_Space[oct][sc+1]; // current�ӵڶ��㿪ʼȡֵ
            down    = DoG_Space[oct][sc+2];
            int sx = current.rows; int sy = current.cols;

			// Ѱ�Ҽ���ֵ��
			// ��ͬ���8������бȽ�
            local_maxima = (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(0,sx-2),Range(0,sy-2)) + 0) & // ��ǰ��������ϵ�
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(0,sx-2),Range(1,sy-1)) + 0) & // ��ǰ��������
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(0,sx-2),Range(2,sy  )) + 0) & // ��ǰ��������µ�
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(1,sx-1),Range(0,sy-2)) + 0) & // ��ǰ������ϵ�
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(1,sx-1),Range(2,sy  )) + 0) & // ��ǰ������µ�
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(2,sx  ),Range(0,sy-2)) + 0) & // ��ǰ��������ϵ�
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(2,sx  ),Range(1,sy-1)) + 0) & // ��ǰ������ҵ�
                           (current(Range(1,sx-1),Range(1,sy-1)) > current(Range(2,sx  ),Range(2,sy  )) + 0) ; // ��ǰ��������µ�

            // ���ϲ��9������бȽ�
            local_maxima = local_maxima & (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(0,sx-2),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(0,sx-2),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(0,sx-2),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(1,sx-1),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(1,sx-1),Range(1,sy-1))) &  // same pixel in top
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(1,sx-1),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(2,sx  ),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(2,sx  ),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > top(Range(2,sx  ),Range(2,sy  )));

			// ���²��9������бȽ�
            local_maxima = local_maxima & (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(0,sx-2),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(0,sx-2),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(0,sx-2),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(1,sx-1),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(1,sx-1),Range(1,sy-1))) &  // same pixel in down
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(1,sx-1),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(2,sx  ),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(2,sx  ),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) > down(Range(2,sx  ),Range(2,sy  )));

			// Ѱ�Ҽ�Сֵ��
			// ��ͬ���8������бȽ�
            local_minima = (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(0,sx-2),Range(0,sy-2))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(0,sx-2),Range(1,sy-1))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(0,sx-2),Range(2,sy  ))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(1,sx-1),Range(0,sy-2))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(1,sx-1),Range(2,sy  ))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(2,sx  ),Range(0,sy-2))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(2,sx  ),Range(1,sy-1))) &
                           (current(Range(1,sx-1),Range(1,sy-1)) < current(Range(2,sx  ),Range(2,sy  ))) ;

			// ���ϲ��9������бȽ�
            local_minima = local_minima & (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(0,sx-2),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(0,sx-2),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(0,sx-2),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(1,sx-1),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(1,sx-1),Range(1,sy-1))) &  // same pixel in top
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(1,sx-1),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(2,sx  ),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(2,sx  ),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < top(Range(2,sx  ),Range(2,sy  )));

			// ���²��9������бȽ�
            local_minima = local_minima & (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(0,sx-2),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(0,sx-2),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(0,sx-2),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(1,sx-1),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(1,sx-1),Range(1,sy-1))) &  // same pixel in down
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(1,sx-1),Range(2,sy  ))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(2,sx  ),Range(0,sy-2))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(2,sx  ),Range(1,sy-1))) &
                                          (current(Range(1,sx-1),Range(1,sy-1)) < down(Range(2,sx  ),Range(2,sy  )));

			// ���㼫��ֵ����߼�Сֵ����������Ϊ�Ǽ�ֵ��
            extrema = local_maxima | local_minima;

            // ��ֵ����Scale_Space�����˱�Ե����Ȧ���أ����丳ֵ����ȷ������
            extrema.copyTo(DoG_Keypts[oct][sc](Range(1, DoG_Keypts[oct][sc].rows-1), Range(1, DoG_Keypts[oct][sc].cols-1)));
        }
    }
}

// DoG�㼫ֵ����ˣ����˵ͷ����ͱ�Ե�㣩
void sift::filter_DoG_extrema()
{
    cout << "\n\n��ֵ�㣨�ͷ����/��Ե�㣩�˳�...\n\n";
    Mat locs; int x, y, rx, ry, fxx, fxy, fyy, deter;
    float trace, curvature;
    float curv_threshold = ((R_THRESHOLD+1)*(R_THRESHOLD+1))/R_THRESHOLD;

	// �����趨һ�������ֵ����������ȡ����
	//curv_threshold = 100;

    for (int oct = 0; oct < num_octaves; oct++)
    {
        for (int sc = 0; sc < num_scales; sc++)
        {
            int reject_contrast_count = 0, reject_edge_count = 0;
			cv::findNonZero(DoG_Keypts[oct][sc], locs);       // �ؼ����λ�ã�x,y���꣩

			// �ж�û����ȡ������������û����ȡ���㣬����������һ��ĺ���������û�б�Ҫ���Ѿ����Լ����findNonZero������opencv�Դ��ĺ�����
			/*
			if (locs.rows == 1)
			{
				continue;
			}
			*/

            int num_keypts = locs.rows;                       // �ؼ�������
            Mat_<uchar> current = DoG_Space[oct][sc+1];

            for (int k = 0; k < num_keypts; k++)
            {
				// �Ĺ�x,y��x��ʾ�кţ�y��ʾ�к�
                y = locs.at<int>(k,0);
                x = locs.at<int>(k,1);

				// ������ȷ���AssignOrientation�����к��ʣ���Լ��
				if (x <= 8 || y <= 8 || x >= DoG_Keypts[oct][sc].rows - 8 || y >= DoG_Keypts[oct][sc].cols - 8)
				{
					DoG_Keypts[oct][sc].at<uchar>(x, y) = 0;
				}

                // ��ȥ�ͷ���ĵ㣬ͨ���Ѹõ���DoG_Keypts�е�ǿ����Ϊ0
                else if (abs(current(x,y)) < CONTRAST_THRESHOLD)
                {
                    DoG_Keypts[oct][sc].at<uchar>(x,y) = 0; 
                    reject_contrast_count++;
                }
                // ��ȥ��Ե��
                else
                {
                    //rx = x+1;
                    //ry = y+1;   
					rx = x;
					ry = y;	// ��ԵһȦ�����Ѿ��������ˣ����õ���Խ��

                    // ��ȡHessian����
                    fxx = current(rx-1,ry) + current(rx+1,ry) - 2*current(rx,ry);   // x����Ķ��ײ��
                    fyy = current(rx,ry-1) + current(rx,ry+1) - 2*current(rx,ry);   // y����Ķ��ײ��
                    fxy = current(rx-1,ry-1) + current(rx+1,ry+1) - current(rx-1,ry+1) - current(rx+1,ry-1); // x,y��45�㣩����Ĳ��
                    // ��Hessian����ļ�������ʽ
                    trace = (float)(fxx + fyy);
                    deter = (fxx*fyy) - (fxy*fxy);
                    curvature = (float)(trace*trace/deter);

					// �������������޳���Ե�㣬ͨ���Ѹõ���DoG_Keypts�е�ǿ����Ϊ0
                    if (deter < 0 || curvature > curv_threshold)  
                    {
                        DoG_Keypts[oct][sc].at<uchar>(x,y) = 0;
                        reject_edge_count++;
                    }

					/*
					if (num_keypts - reject_contrast_count - reject_edge_count > 1)
					{
						// �����ŵ�������浽KeyPoints��ȥ
						KeyPoint keypoint;
						keypoint.pt.x = y;		
						keypoint.pt.y = x;
						//keypoint.id = keypt_id++;
						KeyPoints.push_back(keypoint);
					}	
					*/  // �������������KeyPoints����Ϊ��Ϊ0������Ҳ����������
                }
            }
            printf("\tOctave : %d  Scale : %d  Rejected_Keypoints : %5d  Left_Keypoints : %4d\n", oct+1, sc+1, reject_contrast_count + reject_edge_count, num_keypts - reject_contrast_count - reject_edge_count);
        }
    }
}

// �ؼ��㷽����䣨��ʵ����������л�����Щ��Ĺ�����
void sift::assign_orientation()
{
    cout << "\n�ؼ��㷽�����...\n\n";

    for (int oct = 0; oct < num_octaves; oct++)
    {
        for (int sc = 0; sc < num_scales; sc++)
        {
			// DOG����ȡ��ؼ����Ժ��û���ˣ�����������ؼ����ƥ�䶼���ڳ߶ȿռ�������
            Mat_<uchar> current = Scale_Space[oct][sc+1];
            Magnitude[oct][sc] = Mat::zeros(current.size(), CV_32FC1);
            Orientation[oct][sc] = Mat::zeros(current.size(), CV_32FC1);
			

			// ��ǰ���������ÿһ���ؾ����ݶȴ�С�ͷ���
            for (int x = 1; x < current.rows-1; x++)
            {
                for (int y = 1; y < current.cols-1; y++)
                {
                    // ����x��y����Ĳ��
                    double dx = current(x+1,y) - current(x-1,y); 
                    double dy = current(x,y+1) - current(x,y-1);

                    // ���㵱ǰ����ݶȷ�ֵ�ͷ���
                    Magnitude[oct][sc].at<float>(x,y)   = sqrt(dx * dx + dy * dy);
                    Orientation[oct][sc].at<float>(x,y) = (atan2(dy, dx) == PI)? -PI : atan2(dy,dx);  // Orientation��ȡֵ��-PI��+PI

//                  printf("\tLocation[%d][%d]: Magnitude = %.3f \t Orientation = %.3f\n", x, y, Magnitude[oct][sc].at<uchar>(x,y), Orientation[oct][sc].at<uchar>(x,y));
//					printf("\tLocation[%d][%d]: Magnitude = %.3f \t Orientation = %.3f\n", x, y, Magnitude[oct][sc].at<double>(x, y), Orientation[oct][sc].at<double>(x, y));
                }
            }

			// �����Ѿ�����ǰoctave�͵�ǰscale������������صķ�ֵ�ͷ���
			// �������ǰoctave�͵�ǰscale�Ĺؼ�������ֵ�ͷ�������ֱ��ͼͳ��������
			
			Mat locs;
			cv::findNonZero(DoG_Keypts[oct][sc], locs);
			int x, y, num_keypts = locs.rows;
			//num_total_keypts += num_keypts; // ��ôͳ�Ƴ�����Զ���10�����ң��ĵ�����ͱ���ؼ����������

			//Descriptors�ڽ�������ÿһ�㶼��һ��Mat�����д洢128ά�����ӣ�����Ϊ��ǰ��ؼ���������Descriptors����洢���в��������
			_Descriptors[oct][sc] = Mat::zeros(num_keypts, 128, CV_32FC1);

			// ֱ��ͼ����
			double* hist = new double[36]; // ͳ��36������ÿ10����һ������
			for (int i = 0; i < 36; i++)   // ��ʼ��hist  
				*(hist + i) = 0.0;


			// ��ǰ��octave��scale�Ĺؼ������
			for (int keypt = 0; keypt < num_keypts; keypt++)
			{
				// ��ǰ�ؼ�������
				y = locs.at<int>(keypt, 0); //�ؼ����x����
				x = locs.at<int>(keypt, 1); //�ؼ����y����

				/*
				// �ų��ڱ�Ե�������㣬��Ϊ��ȡ����8*8����ĻҶ�ʱ�ڴ���ը���Ѿ���ȥ��Ե8Ȧ��������
				if (x <= 8 || y <= 8 || x >= DoG_Keypts[oct][sc].rows - 8 || y >= DoG_Keypts[oct][sc].cols - 8)
				{
					continue;
				}
				*/

				// �����Keypoints����һ��
				KeyPoint keypoint;
				keypoint.pt.x = y;		// !!!����д����????
				keypoint.pt.y = x;
				keypoint.octave = oct;    // ����oct��Ϣ����Ҫ����ΪҪ���յ�ǰ��ı���������ԭͼsize����draw��sc��Ϣ����Ҫ����Ӱ������
				KeyPoints.push_back(keypoint);
				num_total_keypts++;

				// �ڹؼ���16*16�����ڱ�����ķ�ֵ�ͷ������ֱ��ͼͳ��
				// ����������Ȩ���ø�˹ģ��
				double weight, mag;
				int bar;

				for (int i = -8; i <= 8; i++)
				{
					for (int j = -8; j <= 8; j++)
					{

						double ri = (double(abs(i)) + 0.000000001) / 8.0001;  //��i,j���򻯵�(0,1)��Χ
						double rj = (double(abs(j)) + 0.000000001) / 8.0001;
						double r = sqrt(ri*ri + rj*rj);
						weight = (1 / sqrt(2 * PI)) * exp(-(r*r) / 2); // ����ɢ��˹�����������Ȩ

						mag = Magnitude[oct][sc].at<float>(x + i, y + j);
						bar = cvRound(36 * (PI - Orientation[oct][sc].at<float>(x + i, y + j)) / (2.0 * PI));
						//printf("%d\n", bar);
						hist[bar] += mag * weight;
					}
				}
				
				//��ֱ��ͼ���и�˹ƽ��
				double prev = hist[36 - 1];
				double temp;
				double h0 = hist[0];

				for (int i = 0; i < 36; i++)
				{
					temp = hist[i];
					hist[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * (i + 1 >= 36 ? h0 : hist[i + 1]);//�Է���ֱ��ͼ���и�˹ƽ��  
					prev = temp;
				}

				// ͳ�Ƴ�ֱ��ͼ����������
				double maxd = hist[0];
				int maxi = 0;
				for (int i = 1; i < 36; i++)
				{
					if (hist[i] > maxd)
					{					   						   //��ȡ36�����е�����ֵ  
						maxd = hist[i];							   //�����ֵ
						maxi = i;								   //���ǵڼ�������
					}
				}

				// ����ؼ����µ�������
				double newKeyptAngle = (maxi - 18) * 10;

				// ͼ����ת����ת��������
				// ������ȫͼ��ת��TODO: ��ȡ����ͼ����ת��������Ч��
				// �õ�ͼ���С
				int width = current.cols;
				int height = current.rows;

				//����ͼ�����ĵ�
				Point2f center;
				center.x = y;
				center.y = x;

				//�����ת�任����
				double scale = 1.0;
				Mat transMat = getRotationMatrix2D(center, newKeyptAngle, scale); //newKeyptAngle����Ϊ����ʾ˳ʱ����ת

				// ��������ת��������? ���Ŀǰ������������90�㣬Ӧ�ð�ͼ����ʱ����ת90��? �ǵ�
				// Ӧ���ǽ�����������˳Ӧ

				// ��ת�����
				Mat rotated;

				//����任
				warpAffine(current, rotated, transMat, Size(width, height));

				//������ǰ�ؼ���������ӣ���������Ķ�̬�������Mat
				//��16*16�����ش��ڣ��������������ĵĹؼ��㣩
				//ע��current�Ѿ���ת�������ˣ�Ӧ�ö�current_rotated���д���
				//��Ϊ�����Źؼ��������ת�ģ���ͼ��ߴ�û�иı䣬�ؼ������������к�û�иı䣬����x,y
				//��Ҫ���и�˹��Ȩ��ֱ���þ���ؼ���ľ�����ΪȨ
				
				//����16�����ӵ�
				//��ÿ���ؼ��㣬����1��128ά�����ӣ���16�����ӵ㣬ÿ�����ӵ�8������

				double descriptor[16][8];
				int seedSeq = 0; //��¼�������ڼ������ӵ�

				for (int seedi = -6; seedi <= 6; seedi += 4)
				{
					for (int seedj = -6; seedj <= 6; seedj += 4)
					{
						//��ÿ�����ӵ��ڱ���
						//double* seedHist = new double[8]; // ÿ�����ӵ��ֱ��ͼ��ͳ��8������
						double* seedHist = descriptor[seedSeq]; // ��ǰ���ӵ�

						int seedBar;
						for (int i = 0; i < 8; i++)       // ��ʼ��seedHist  
							*(seedHist + i) = 0.0;

						for (int i = -2; i < 2; i++)
						{
							for (int j = -2; j < 2; j++)
							{
								float dx = rotated.at<uchar>(x + seedi + i + 1, y + seedj + j) - rotated.at<uchar>(x + seedi + i - 1, y + seedj + j);
								float dy = rotated.at<uchar>(x + seedi + i, y + seedj + j + 1) - rotated.at<uchar>(x + seedi + i, y + seedj + j - 1);

								float eachMeg = sqrt(dx * dx + dy * dy);
								float eachOri = (atan2(dy, dx) == PI) ? -PI : atan2(dy, dx);
								double eachx = (double(abs(seedi + i)) + 0.000000001) / 8.0001;  //����ά������򻯵�(0,1)��Χ
								double eachy = (double(abs(seedj + j)) + 0.000000001) / 8.0001;
								double eachDist = sqrt(eachx*eachx + eachy*eachy);				//��ǰ������ؼ������صľ���
								float eachWeight = (1 / sqrt(2 * PI)) * exp(-(eachDist*eachDist) / 2); //���˹Ȩֵ
								seedBar = cvRound(8 * (PI - eachOri) / (2.0 * PI));
								seedHist[seedBar] += eachMeg * eachWeight;
							}
						}
						seedSeq++;
					}
				}

				//��ǰ�ؼ�����Χ16�����ӵ��ֱ��ͼ��128ά�����ӣ��������
				//ֱ��ͼ������descriptor[16][8]������
				//���������԰ѵ�ǰdescriptor[16][8]�浽һ����Mat(Descriptors)�У���Mat��˳��洢���йؼ����������
				//Descriptors�ڽ�������ÿһ�㶼��һ��Mat�����д洢128ά�����ӣ�����Ϊ��ǰ��ؼ���������Descriptors����洢���в��������

				// ֱ��ͼ�ضϣ���һ������������Ӱ��
				for (int v = 0; v < 128; v++)
				{

					double test1 = *(descriptor[0] + v);
					if (*(descriptor[0] + v) > 60)
					{
						double test2 = *(descriptor[0] + v);
						*(descriptor[0] + v) = 60;
					}
				}

				// ֱ��ͼ��һ��
				double cur, len_inv, len_sq = 0.00001;

				for (int v = 0; v < 128; v++)
				{
					cur = *(descriptor[0] + v);
					len_sq += cur*cur;
				}

				len_inv = 1.0 / sqrt(len_sq);

				for (int v = 0; v < 128; v++)
				{
					// �Ծ��ڹ�һ��
					*(descriptor[0] + v) *= len_inv;
					// ��ֱ��ͼ��Ϊ�����Ӵ���_Descriptors
					_Descriptors[oct][sc].at<float>(keypt, v) = *(descriptor[0] + v); //descriptor[0]��128ά�����е�һ��Ԫ�صĵ�ַ
				}
				
			} // ��ǰ��ؼ����������
        }
    } // ͼ����������в�����������������Ѿ�����_Descriptors


}


// ��ȡ�ؼ���
vector<KeyPoint> sift::get_keypoints()
{
	return KeyPoints;
}


// ��ȡ128ά������
Mat sift::get_descriptors()
{
	// ��ʼ��Descriptors��
	Descriptors = Mat(num_total_keypts, 128, CV_32FC1);
	float* descriptorData = Descriptors.ptr<float>(0);
	int globalKeypt = 0; // Ϊ��ȷ�����뵽Descriptors����һ�У���Ҫ����һ��ȫ�ֵĹؼ��������������ǵ�ǰoctave��scale������

	// ��_Descriptor�е������Ӹ�����Mat��Descriptors������_Descriptor��ÿһ��
	for (int oct = 0; oct < num_octaves; oct++)
	{
		for (int sc = 0; sc < num_scales; sc++)
		{
			// ����ͨ������Mat��Ԫ�ؽ��в���
			// ��for������ɸѡ��_Descriptors��ֻ��1���ؼ���Ĳ㣨��ʵ��û�йؼ��㣩��������Щ�ٹؼ�����뵽Descriptors��
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
	// �����б�Ҫ��_Descriptorsת��Descriptors�ģ�_Descriptors�к���[û�йؼ���Ĳ�]�����Ҳ���Descriptors��KeyPoints�Ķ�Ӧ��ϵ��
	return Descriptors;
}

// Ӱ����ʾ����
void sift::display_images(string left_or_right, int oct, int sc, int mode)
{
    stringstream ss_oct, ss_sc;
    ss_oct << oct; ss_sc << sc;

    switch (mode)
    {
        case 1:        // ��ʾ�߶ȿռ�ͼ��
        {
        string wSS = left_or_right + " SS Octave " + ss_oct.str() + " Scale " + ss_sc.str();
            imshow(wSS, Scale_Space[oct-1][sc-1]);
            break;
        }
        case 2:        // ��ʾDoGͼ��
        {
            string wDoG = left_or_right + " DoG_Space Octave " + ss_oct.str() + " Scale " + ss_sc.str();
            imshow(wDoG, DoG_Space[oct-1][sc-1]);
            break;
        }
        case 3:       // ��ʾDoG�ؼ���
        {
            string wDoGKP = left_or_right + " DoG_Space Keypts Octave " + ss_oct.str() + " Scale " + ss_sc.str();
            imshow(wDoGKP, DoG_Keypts[oct-1][sc-1]);
            break;
        }
    }
}


// �Լ����findNonZero�������������°汾��opencv(2.4.13)���ҵ��˿�ֱ�ӵ��õĺ���
/*
Mat sift::findNonZero(Mat _src)
{
	//CV_INSTRUMENT_REGION();

	//Mat src = _src.getMat();
	Mat src, _idx;
	_src.copyTo(src);


	// ɾ����ͼ���Ե��8�������ڣ��ķ���㣬�����Ϊ0
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
	_idx.create(n, 1, CV_32SC2); //��ͨ��

	//CV_Assert(idx.isContinuous());
	Point* idx_ptr = _idx.ptr<Point>();


	for (int i = 0; i < src.rows - 1; i++)
	{
		const uchar* bin_ptr = src.ptr(i);
		for (int j = 0; j < src.cols - 1; j++)
			if (bin_ptr[j])
			{
				*idx_ptr++ = Point(j, i);
				// ��ָ��_idx��ָ���ƺ��Ĳ���_idx������ֵ�����ԣ�ֱ����Mat���ָ���ֵ
			}
	}
	//*idx_ptr++ = Point(0, 0);
	return _idx;
}
*/