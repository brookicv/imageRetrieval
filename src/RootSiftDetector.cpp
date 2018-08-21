
#include "RootSiftDetector.h"

RootSiftDetector::RootSiftDetector(){

    m_contrastThreshold         = 0.04;
    m_edgeThreshold             = 10;
    m_keypointCountThreshold    = 10;

    m_siftDetector = cv::xfeatures2d::SIFT::create();
}

RootSiftDetector::~RootSiftDetector(){}



RootSiftDetector::RootSiftDetector(double edgeThreshold,double contrastThreshold,int keypointCountThreshold) {
    m_contrastThreshold         = contrastThreshold;
    m_edgeThreshold             = edgeThreshold;
    m_keypointCountThreshold    = keypointCountThreshold;

    m_siftDetector = cv::xfeatures2d::SIFT::create(0,3,m_contrastThreshold,m_edgeThreshold);
}

void RootSiftDetector::reset(double edgeThreshold,double contrastThreshold){

    m_contrastThreshold = contrastThreshold;
    m_edgeThreshold = edgeThreshold;

    m_siftDetector = cv::xfeatures2d::SIFT::create(0,3,m_contrastThreshold,m_edgeThreshold);
}


void RootSiftDetector::rootSift(const cv::Mat &siftFeature,cv::Mat &rootSiftFeature) {

    for(int i = 0; i < siftFeature.rows; i ++){
        // Conver to float type
        Mat f;
        siftFeature.row(i).convertTo(f,CV_32FC1);

        normalize(f,f,1,0,NORM_L1); // l1 normalize
        sqrt(f,f); // sqrt-root  root-sift
        rootSiftFeature.push_back(f);
    }

}

void RootSiftDetector::detectAndCompute(const cv::Mat &img,std::vector<cv::KeyPoint> &kpts,cv::Mat &rootSift){

    cv::Mat features;
    m_siftDetector->detectAndCompute(img,cv::noArray(),kpts,features);

    if(kpts.size() < m_keypointCountThreshold){
        reset();
        m_siftDetector->detectAndCompute(img,cv::noArray(),kpts,features);
    }

    rootSift(features,rootSift);
}