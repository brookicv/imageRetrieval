
/*
 * File: Utility.h
 * Date: 2018/8/6
 * Author: LiQiang
 * Description: Utility functions
 * 
*/

#ifndef H__UTILITY__
#define H__UTILITY__

#include <vector>
#include <opencv2/opencv.hpp>
#include <string>

/*
    Warp sfit feature detector and transform sift to root-sift
*/
class siftDetecotor{

public:
    void static extractFeatures(const std::vector<std::string> &imageFileList,std::vector<std::vector<cv::KeyPoint>> &kptList,std::vector<cv::Mat> &features);
    void static extractFeatures(const std::string &file,std::vector<cv::KeyPoint> &kpts,cv::Mat &feature);

    void static extractFeaturesFromImg(const cv::Mat &img,std::vector<cv::KeyPoint> &kpts,cv::Mat &feature);

    void static rootSift(const cv::Mat &siftFeature,cv::Mat &rootSiftFeature);
    void static rootSift(const std::vector<cv::Mat> &siftFeatures,std::vector<cv::Mat> &rootSiftFeatures);
};

class PathManager{

public:
    void static get_path_list(const std::string &folder,std::vector<std::string> &file_list);
    void static extractFilename(const std::string &path,std::string &name);
    std::string static contact(const std::string filename,std::initializer_list<std::string> folders);
};

#endif