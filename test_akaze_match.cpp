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
    double phai  = pt.y / (double)h * M_PI;
    pnt1.x = sin( phai ) * cos( theta );
    pnt1.y = cos( phai );
    pnt1.z = sin( phai ) * sin( theta );
    return pnt1;
}

int main(int argc, char** argv )
{
    cv::Mat img1, img2;
    img1 = cv::imread( argv[1], 1 );
    if ( !img1.data )
    {
        printf("No image data \n");
        return -1;
    }
    img2 = cv::imread( argv[2], 1 );
    if ( !img2.data )
    {
        printf("No image 2 data \n");
        return -1;
    }
    inlier_threshold = std::atof( argv[3] );
//    Mat img1 = imread("graf1.png", IMREAD_GRAYSCALE);
//    Mat img2 = imread("graf3.png", IMREAD_GRAYSCALE);

    cv::Size imgSz = img1.size();
    w = imgSz.width;
    h = imgSz.height;

    Mat homography;
    FileStorage fs("H1to3p.xml", FileStorage::READ);
    fs.getFirstTopLevelNode() >> homography;

    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    AKAZE akaze;
    akaze(img1, noArray(), kpts1, desc1);
    akaze(img2, noArray(), kpts2, desc2);

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
        cout << pnt1 << ":" << pnt2 << "\n";
        spherePoint1.push_back( pnt1 );
        spherePoint2.push_back( pnt2 );
    }
/*
0.9559639913966587, -0.1512785456133775, 0.008659000430174126, -0.03115525448941475;
 0.1213505258011772, 0.9521654706553296, 0.1884811247447123, -0.02286234084694982;
 -0.02882466337354708, -0.191771707675626, 0.9698523604117755, 0.003293610165961219
     mat.m00 = 0.9559639913966587f;
     mat.m01 =  0.1213505258011772f;
     mat.m02 =  -0.02882466337354708f;
		mat.m03 = 0.0f;
		mat.m10 = -0.1512785456133775f;
		mat.m11 = 0.9521654706553296f;
		mat.m12 = -0.191771707675626f;
		mat.m13 = 0.0f;
		mat.m20 = 0.008659000430174126;
		mat.m21 = 0.1884811247447123;
		mat.m22 = 0.9698523604117755;
		mat.m23 = 0.0f;
		mat.m30 =  -0.03115525448941475f;
		mat.m31 =  -0.02286234084694982f;
		mat.m32 =  0.00329361016596121f;
		mat.m33 = 1.0f;
*/
//    spherePoint1.push_back( Point3f( 0.7071067, 0, 0.7071067 ) );
//    spherePoint2.push_back( Point3f( 0, 0, 1 ) );
//    spherePoint1.push_back( Point3f( 0, 0, 1 ) );
//    spherePoint2.push_back( Point3f( -0.7071067, 0, 0.7071067 ) );
//    spherePoint1.push_back( Point3f( -1, 0, 0 ) );
//    spherePoint2.push_back( Point3f( -0.7071067, 0, -0.7071067 ) );
//    spherePoint1.push_back( Point3f( -0.7071067, 0, -0.7071067 ) );
//    spherePoint2.push_back( Point3f( 0, 0, -1 ) );

    Mat estimateMat;
    vector<uchar> outliers;
    estimateAffine3D( spherePoint1, spherePoint2, estimateMat, outliers, 2.0, 0.8 );
    cout << estimateMat << "\n";
    cout << outliers.size()  << "\n";

    Mat res;
    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
    imwrite("res.png", res);

    double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
    cout << "A-KAZE Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << matched1.size() << endl;
    cout << "# Inliers:                            \t" << inliers1.size() << endl;
    cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
    cout << endl;

    return 0;
}
