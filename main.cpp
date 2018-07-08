#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <string>
extern "C" {
    #include <vl/kmeans.h>
}

#include "vl_sift_detetor.h"

using namespace std;
using namespace cv;

void sift_test(){
    const string file = "../1.jpg";
    Mat img = imread(file,IMREAD_GRAYSCALE);
    Mat color_img = imread(file);
    Mat float_img;
    img.convertTo(float_img,CV_32F);

    int rows = img.rows;
    int cols = img.cols;
    VlSiftFilt* vl_sift =  vl_sift_new(cols,rows,-1,3,0);
    vl_sift_set_peak_thresh(vl_sift,0.04);
    vl_sift_set_edge_thresh(vl_sift,10);

    vl_sift_pix *data = (vl_sift_pix*)(float_img.data);


    vector<VlSiftKeypoint> kpts;
    vector<float*> descriptors;

    vl_sift_extract(vl_sift,data,kpts,descriptors);

    vl_sift_delete(vl_sift);

    for(int i = 0; i < kpts.size(); i ++) {
        circle(color_img,Point(kpts[i].x,kpts[i].y),kpts[i].sigma,Scalar(0,255,0));
    }

    vector<KeyPoint> oc_kpts;
    Mat oc_des;
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    sift->detectAndCompute(color_img,noArray(),oc_kpts,oc_des);

    for(int i = 0; i < oc_kpts.size(); i ++) {
        circle(color_img,oc_kpts[i].pt,oc_kpts[i].size,Scalar(255,0,0));
    }

    cout << "vl_sift:" << descriptors.size() << endl;
    cout << "opencv_sift-descriptors:" << oc_des.size() << endl;
    
    namedWindow("SIFT");
    imshow("SIFT",color_img);
    waitKey();
}


int main() 
{
    
    return 0;
}