#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <string>
extern "C" {
    #include <vl/sift.h>
    #include <vl/kmeans.h>
}

using namespace std;
using namespace cv;

void bow()
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

/*
    Extract sift using vlfeat
    parameters:
        vl_sfit, VlSiftFilt* 
        data , image pixel data ,to be convert to float
        kpts, keypoint list
        descriptors, descriptor. Need to free the memory after using.
*/
void vl_sift_extract(VlSiftFilt *vl_sift, vl_sift_pix* data,
                    vector<VlSiftKeypoint> &kpts,vector<float*> &descriptors) {
    
    // Detect keypoint and compute descriptor in each octave
    if(vl_sift_process_first_octave(vl_sift,data) != VL_ERR_EOF){
        while(true){
            vl_sift_detect(vl_sift);

            VlSiftKeypoint* pKpts = vl_sift->keys;
            for(int i = 0; i < vl_sift->nkeys; i ++) {

                double angles[4];
                // 计算特征点的方向，包括主方向和辅方向，最多4个
                int angleCount = vl_sift_calc_keypoint_orientations(vl_sift,angles,pKpts);

                // 对于方向多于一个的特征点，每个方向分别计算特征描述符
                // 并且将特征点复制多个
                for(int i = 0 ; i < angleCount; i ++){
                    float *des = new float[128];
                    vl_sift_calc_keypoint_descriptor(vl_sift,des,pKpts,angles[0]);
                    descriptors.push_back(des);
                    kpts.push_back(*pKpts);
                }
                pKpts ++;
            }    
            // Process next octave
            if(vl_sift_process_next_octave(vl_sift) == VL_ERR_EOF) {
                break ;
            }
        }
    }
}

void convert_sift_to_root_sift(vector<float*> &descriptors){
        /*
        Trans sift to rootSift
        1. L1 normalize each descriptor(128-dims vector)
        2. sqrt-root each element
        3. [option] L2-normalize descriptor vector
     */
    for(float* ptr : descriptors) {
        float sum = 0;
        for(int i = 0; i < 128; i ++)
            sum += *(ptr + i);
        for(int i = 0; i < 128; i ++){
            *(ptr + i) /= sum;
            *(ptr + i) = sqrt(*(ptr + i));
        }
    }
}

void vl_root_sift_extract(VlSiftFilt *vl_sift, vl_sift_pix* data,vector<VlSiftKeypoint> &kpts,vector<float*> &descriptors) {

    vl_sift_extract(vl_sift,data,kpts,descriptors);
    convert_sift_to_root_sift(descriptors);
}



int main() 
{
    const string file = "../0.jpg";
    Mat img = imread(file,IMREAD_GRAYSCALE);
    Mat color_img = imread(file);
    Mat float_img;
    img.convertTo(float_img,CV_32F);

    int rows = img.rows;
    int cols = img.cols;
    VlSiftFilt* vl_sift =  vl_sift_new(cols,rows,4,3,0);
    vl_sift_set_peak_thresh(vl_sift,4);
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
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(427);
    sift->detectAndCompute(color_img,noArray(),oc_kpts,oc_des);

    for(int i = 0; i < oc_kpts.size(); i ++) {
        circle(color_img,oc_kpts[i].pt,oc_kpts[i].size,Scalar(255,0,0));
    }

    cout << "vl_sift:" << descriptors.size() << endl;
    cout << "opencv_sift:" << oc_des.size() << endl;
    

    namedWindow("SIFT");
    imshow("SIFT",color_img);
    waitKey();
    return 0;


}