#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <string>

using namespace std;
using namespace cv;

int main() 
{
    const string file = "";
    Mat img = imread(file);

    vector<KeyPoint> kpts;
    Mat descriptors;

    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    sift->detectAndCompute(img,noArray(),kpts,descriptors);

    // Trans sift to rootSift
    for(int y = 0 ; y < descriptors.rows; y ++) {
        for(int x = 0;  x < descriptors.cols; x ++) {
            descriptors.at<float>(y,x) = sqrt(descriptors.at<float>(y,x));
        }
    }

    float threshold = pow(10,12);
    // L2 normalization
    for(int y = 0; y < descriptors.rows; y ++) {
        float sum = 0;
        for(int x = 0; x < descriptors.cols; x ++) {
            sum += descriptors.at<float>(y,x) * descriptors.at<float>(y,x);
        }
    }
}