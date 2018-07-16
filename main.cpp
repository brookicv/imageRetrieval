#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <opencv2/flann.hpp>

#include "DBoW3.h"
#include "utils.h"

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

void extract_features(vector<string> image_file_list,vector<Mat> features){

    int index = 1;
    int count = 0;
    for(const string &str : image_file_list){

        auto img = imread(str,IMREAD_GRAYSCALE);
        if(img.empty()){
            cerr << "Open image #" << str << " features failed" << endl;
            continue;
        }

        cout << "Extract #" << index << "st image #" << str << " features" << endl;
        auto sift = xfeatures2d::SIFT::create(0,3,0.2,10);
        vector<KeyPoint> kpts;
        Mat des;
        sift->detectAndCompute(img,noArray(),kpts,des);
        features.emplace_back(des); 
        count += des.rows;
        index ++ ;
    }
    
    cout << "Extract #" << index << "# images features done!" << "Count of features:#" << count << endl;

}

void vocabulary(const vector<Mat> &features){
    //Branching factor and depth levels
    const int K = 9;
    const int L = 3;
    const DBoW3::WeightingType weight = DBoW3::TF_IDF;
    const DBoW3::ScoringType score = DBoW3::L1_NORM;

    DBoW3::Vocabulary voc(K,L,weight,score);
    cout << "Creating a small " << K << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "...done!" << endl;
    
    cout << "Vocabulary infomation: " << endl << voc << endl << endl;

    cout << "Matching images against themselves (0 low,1 hight): " << endl;
    DBoW3::BowVector v1,v2;
    int i = 0, j = 0;
    for(const Mat &m : features) {
        voc.transform(m,v1);
        for(const Mat &m1 : features){
            voc.transform(m1,v2);
            double score = voc.score(v1,v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
            j ++;
        }
        i ++;
    }

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("small_voc.yml.gz");
    cout << "Done" << endl;
}


int main() 
{
    const string image_folder = "/home/book/git/imageRetrieval/images";
    vector<string> image_file_list;
    get_file_name_list(image_folder,image_file_list);
    vector<Mat> features;
    extract_features(image_file_list,features);
    vocabulary(features);
    return 0;
}