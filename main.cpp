#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <string>
extern "C" {
    #include <vl/kmeans.h>
}

#include "siftDetector.h"

using namespace std;
using namespace cv;

void sift_test(){
    const string file = "../1.jpg";
    Mat img = imread(file);
    SiftDetector sift_detector;

    vector<VlSiftKeypoint> kpts;
    vector<vector<float>> descriptors;
    sift_detector.detect_and_compute(img,kpts,descriptors);

    for(int i = 0; i < kpts.size(); i ++) {
        circle(img,Point(kpts[i].x,kpts[i].y),kpts[i].sigma,Scalar(0,255,0));
    }

    vector<KeyPoint> oc_kpts;
    Mat oc_des;
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    sift->detectAndCompute(img,noArray(),oc_kpts,oc_des);

    for(int i = 0; i < oc_kpts.size(); i ++) {
        circle(img,oc_kpts[i].pt,oc_kpts[i].size,Scalar(255,0,0));
    }

    cout << "vl_sift:" << descriptors.size() << endl;
    cout << "opencv_sift-descriptors:" << oc_des.size() << endl;
    
    namedWindow("SIFT");
    imshow("SIFT",img);
    waitKey();
}


int main() 
{
    sift_test();
    return 0;
}