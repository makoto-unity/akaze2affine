#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;

static float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
static int w;
static int h;

static Point3f getSpherePoint( Point2f pt )
{
    Point3f pnt1;
    double theta = pt.x / (double)w * M_PI * 2.0f;
    double phai  = ((double)h*0.5 - pt.y) / (double)h * M_PI;
    pnt1.x = cos( phai ) * cos( theta );
    pnt1.y = sin( phai );
    pnt1.z = cos( phai ) * sin( theta );
    return pnt1;
}

///////////////////////////////////////////////
// 回転行列→クォータニオン変換
//
// qx, qy, qz, qw : クォータニオン成分（出力）
// m11-m33 : 回転行列成分
//
// ※注意：
// 行列成分はDirectX形式（行方向が軸の向き）です
// OpenGL形式（列方向が軸の向き）の場合は
// 転置した値を入れて下さい。
/*
static bool transformRotMatToQuaternion(
    double &qx, double &qy, double &qz, double &qw,
    Mat &m
) {
//    cout << m.at<double>(0,0) << " , " << m.at<double>(0,1) << " , " << m.at<double>(0,2) <<  endl;
//    cout << m.at<double>(1,0) << " , " << m.at<double>(1,1) << " , " << m.at<double>(1,2) <<  endl;
//    cout << m.at<double>(2,0) << " , " << m.at<double>(2,1) << " , " << m.at<double>(2,2) <<  endl;
    double m11 = m.at<double>(0,0); double m12 = m.at<double>(0,1); double m13 = m.at<double>(0,2);
    double m21 = m.at<double>(1,0); double m22 = m.at<double>(1,1); double m23 = m.at<double>(1,2);
    double m31 = m.at<double>(2,0); double m32 = m.at<double>(2,1); double m33 = m.at<double>(2,2);
    // 最大成分を検索
    double elem[ 4 ]; // 0:x, 1:y, 2:z, 3:w
    elem[ 0 ] = m11 - m22 - m33 + 1.0;
    elem[ 1 ] = -m11 + m22 - m33 + 1.0;
    elem[ 2 ] = -m11 - m22 + m33 + 1.0;
    elem[ 3 ] = m11 + m22 + m33 + 1.0;

    unsigned biggestIndex = 0;
    for ( int i = 1; i < 4; i++ ) {
        if ( elem[i] > elem[biggestIndex] )
            biggestIndex = i;
    }

    if ( elem[biggestIndex] < 0.0 )
        return false; // 引数の行列に間違いあり！

    // 最大要素の値を算出
    double *q[4] = {&qx, &qy, &qz, &qw};
    double v = sqrtf( elem[biggestIndex] ) * 0.5;
    *q[biggestIndex] = v;
    double mult = 0.25 / v;

    switch ( biggestIndex ) {
    case 0: // x
        *q[1] = (m12 + m21) * mult;
        *q[2] = (m31 + m13) * mult;
        *q[3] = (m23 - m32) * mult;
        break;
    case 1: // y
        *q[0] = (m12 + m21) * mult;
        *q[2] = (m23 + m32) * mult;
        *q[3] = (m31 - m13) * mult;
        break;
    case 2: // z
        *q[0] = (m31 + m13) * mult;
        *q[1] = (m23 + m32) * mult;
        *q[3] = (m12 - m21) * mult;
    break;
    case 3: // w
        *q[0] = (m23 - m32) * mult;
        *q[1] = (m31 - m13) * mult;
        *q[2] = (m12 - m21) * mult;
        break;
    }

    return true;
}
*/

static bool transformRotMatToQuaternion(
    double &qx, double &qy, double &qz, double &qw,
    Mat &m
    )
{
    double m11 = m.at<double>(0,0); double m12 = m.at<double>(1,0); double m13 = m.at<double>(2,0);
    double m21 = m.at<double>(0,1); double m22 = m.at<double>(1,1); double m23 = m.at<double>(2,1);
    double m31 = m.at<double>(0,2); double m32 = m.at<double>(1,2); double m33 = m.at<double>(2,2);
    double T = 1 + m11 + m22 + m33;
		
    if ( T > 0.00000001 ){
        double S = sqrt(T) * 2.0;
        qx = ( m32 - m23 ) / S;
        qy = ( m13 - m31 ) / S;
        qz = ( m21 - m12 ) / S;
        qw = 0.25 * S;
    }else{
        return false;
    }
		
    return true;
}

int main(int argc, char** argv )
{
//    cv::Mat img1, img2;
//    img1 = cv::imread( argv[1], 1 );
//    if ( !img1.data )
//    {
//        printf("No image data \n");
//        return -1;
//    }
//    img2 = cv::imread( argv[2], 1 );
//    if ( !img2.data )
//    {
//        printf("No image 2 data \n");
//        return -1;
//    }
//    inlier_threshold = std::atof( argv[3] );
//    Mat img1 = imread("graf1.png", IMREAD_GRAYSCALE);
//    Mat img2 = imread("graf3.png", IMREAD_GRAYSCALE);
	Mat prev;

    std::string filePath = argv[1];
    //VideoCapture capture = VideoCapture(argv[1]);

    int fromId = std::atoi(argv[2]);
    int toId   = std::atoi(argv[3]);

/*
    if ( fromId > 0 ) {
        for( int i = 0; i<fromId ; i++ ) {
            capture >> prev;
        }
    } else {
        cout << "0,0,0,0,1" << endl;
        capture >> prev;
    }
*/
    char charId0[20];
    sprintf(charId0, "%05d", fromId);
    string strId0(charId0);
    std::string fileName0 = filePath + strId0 + ".tif";
	prev = cv::imread( fileName0, 1 );

    inlier_threshold = std::atof( argv[4] );

    cv::Size imgSz = prev.size();
    w = imgSz.width;
    h = imgSz.height;

    Mat homography;
    FileStorage fs("H1to3p.xml", FileStorage::READ);
    fs.getFirstTopLevelNode() >> homography;

    //while (waitKey(1) == -1) 
    //for( int i=0 ; i<10 ; i++ ) 
    for( int i=fromId+1 ; i<=toId ; i++ )
    {
        char charId[20];
        sprintf(charId, "%05d", i);
        string strId(charId);
        std::string fileName = filePath + strId + ".tif";
//        cout << fileName << endl;
        // 現在のフレームを保存
		Mat curr;
        curr = cv::imread( fileName, IMREAD_GRAYSCALE );
		//capture >> curr;

        vector<KeyPoint> kpts1, kpts2;
        Mat desc1, desc2;

        AKAZE akaze;
        akaze(prev, noArray(), kpts1, desc1);
        akaze(curr, noArray(), kpts2, desc2);

        BFMatcher matcher(NORM_HAMMING);
        vector< vector<DMatch> > nn_matches;
        matcher.knnMatch(desc1, desc2, nn_matches, 2);

        vector<KeyPoint> matched1, matched2, inliers1, inliers2;
        vector<DMatch> good_matches;
        for(size_t i = 0; i < nn_matches.size(); i++) {
            DMatch first = nn_matches[i][0];
            float dist1 = nn_matches[i][0].distance;
            float dist2 = nn_matches[i][1].distance;

            if(dist1 < nn_match_ratio * dist2) {
                matched1.push_back(kpts1[first.queryIdx]);
                matched2.push_back(kpts2[first.trainIdx]);
            }
        }

        std::vector< cv::Vec2f > points1( matched1.size());
        std::vector< cv::Vec2f > points2( matched1.size());
        for(unsigned i = 0; i < matched1.size(); i++) {
            points1[i][0] = matched1[i].pt.x;
            points1[i][1] = matched1[i].pt.y;
            points2[i][0] = matched2[i].pt.x;
            points2[i][1] = matched2[i].pt.y;
        }
        homography = cv::findHomography( points1, points2, 0, 5.0);

        for(unsigned i = 0; i < matched1.size(); i++) {
            Mat col = Mat::ones(3, 1, CV_64F);
            col.at<double>(0) = matched1[i].pt.x;
            col.at<double>(1) = matched1[i].pt.y;

            col = homography * col;
            col /= col.at<double>(2);
            double dist = sqrt( pow(col.at<double>(0) - matched2[i].pt.x, 2) +
                                pow(col.at<double>(1) - matched2[i].pt.y, 2));

            if(dist < inlier_threshold) {
                int new_i = static_cast<int>(inliers1.size());
                inliers1.push_back(matched1[i]);
                inliers2.push_back(matched2[i]);
                good_matches.push_back(DMatch(new_i, new_i, 0));
            }
        }

        vector<Point3f> spherePoint1, spherePoint2;
        for(unsigned i = 0; i < inliers1.size(); i++) {
            Point3f pnt1 = getSpherePoint( inliers1[i].pt );
            Point3f pnt2 = getSpherePoint( inliers2[i].pt );
            spherePoint1.push_back( pnt1 );
            spherePoint2.push_back( pnt2 );
        }
/*
  0.9559639913966587, -0.1512785456133775, 0.008659000430174126, -0.03115525448941475;
  0.1213505258011772, 0.9521654706553296, 0.1884811247447123, -0.02286234084694982;
  -0.02882466337354708, -0.191771707675626, 0.9698523604117755, 0.003293610165961219

0.9168613690158312, 0.215071121623276, -0.3278535119975714, 0.002336346562723357;
 -0.2419376892706289, 0.9616379242346873, 0.05016308679857526, -0.03482866848896454;
 0.305011184524675, 0.04027020491171077, 0.9560339475879948, -0.01581409040900132

  mat.m00 = 0.9168613690158312f;
  mat.m01 = -0.2419376892706289f;
  mat.m02 = 0.305011184524675f;
  mat.m03 = 0.0f;
  mat.m10 = 0.215071121623276f;
  mat.m11 = 0.9616379242346873f;
  mat.m12 = 0.04027020491171077f;
  mat.m13 = 0.0f;
  mat.m20 = -0.3278535119975714f;
  mat.m21 = 0.05016308679857526f;
  mat.m22 = 0.9560339475879948f;
  mat.m23 = 0.0f;
  mat.m30 = 0.002336346562723357f;
  mat.m31 = -0.03482866848896454f;
  mat.m32 = -0.01581409040900132f;
  mat.m33 = 1.0f;
*/
/*
    spherePoint1.push_back( Point3f( 0.7071067, 0, 0.7071067 ) );
    spherePoint2.push_back( Point3f( 0, 0, 1 ) );
    spherePoint1.push_back( Point3f( 0, 0, 1 ) );
    spherePoint2.push_back( Point3f( -0.7071067, 0, 0.7071067 ) );
    spherePoint1.push_back( Point3f( -1, 0, 0 ) );
    spherePoint2.push_back( Point3f( -0.7071067, 0, -0.7071067 ) );
    spherePoint1.push_back( Point3f( -0.7071067, 0, -0.7071067 ) );
    spherePoint2.push_back( Point3f( 0, 0, -1 ) );
*/
        if ( spherePoint1.size() > 0 ) {
            Mat estimateMat;
            vector<uchar> outliers;
            int ret = estimateAffine3D( spherePoint1, spherePoint2, estimateMat, outliers, 2.0, 0.8 );
//        estimateMat.at<double>(1,1) = 1.0;
//        cout << estimateMat << endl;
//        cout << ret << endl;
//        cout << outliers.size()  << endl;

            if ( ret == 1 ) {
                double qx, qy, qz, qw;
                transformRotMatToQuaternion(qx, qy, qz, qw, estimateMat);
                cout << i << "," << qx << "," << qy << "," << -qz << "," << qw << endl;
            } else {
                cout << i << ",*,*,*,*" <<endl;
            }
        } else {
            cout << i << ",*,*,*,*" <<endl;
        }

//        Mat res;
//        drawMatches(prev, inliers1, curr, inliers2, good_matches, res);
//        imwrite("res.png", res);

//        double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
//        cout << "A-KAZE Matching Results" << endl;
//        cout << "*******************************" << endl;
//        cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
//        cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
//        cout << "# Matches:                            \t" << matched1.size() << endl;
//        cout << "# Inliers:                            \t" << inliers1.size() << endl;
//        cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
//        cout << endl;

		// 前のフレームを保存
		prev = curr;
    }

    return 0;
}
