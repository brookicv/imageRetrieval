/**
 * 
 * Date  : 2018/8/21
 * Author: LiQiang
 * Description:
 *  Wrap up root sift detetor
 * 
 */
#ifndef __H__FEATUREDETECTOR__
#define __H__FEATUREDETECTOR__

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

class RootSiftDetector {
    
    RootSiftDetector();
    ~RootSiftDetector();

    RootSiftDetector(double edgeThreshold,double contrastThreshold,int keypointCountThreshold);

    /*
        Reset sift detector to default or special parameters
    */
    void reset(double edgeThreshold = 10,double contrastThreshold = 0.04);

    /*
        Detect and compute
    */
    void detectAndCompute(const cv::Mat &img,std::vector<cv::KeyPoint> &kpts,cv::Mat &rootSift);

    /*
        Transform sift descriptor to root sift descriptor
    */
    void rootSift(const cv::Mat &siftFeature,cv::Mat &rootSiftFeature);

private:

    /*
        The threshold used to filter out edge-like features. 
        Note that the its meaning is different from the contrastThreshold, 
        i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
    */
    double m_edgeThreshold;


    /*
        The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. 
        The larger the threshold, the less features are produced by the detector
    */ 
    double m_contrastThreshold;

    /*  
        Threshold of keypoints count
        If the count of detected keypoints less than it,re-detect with default sift parameters
    */
    int m_keypointCountThreshold;
    
    /*
        Sift detector
    */
    cv::Ptr<cv::xfeatures2d::SIFT> m_siftDetector;
}

#endif