
/*
    对vl_feat 提供的sift特征检测方法进行的封装，并提供了save和load sift特征的静态方法。
    - Brookicv
    -- 2018/7/7
*/

#ifndef VL_SIFT_DETECTOR_H
#define VL_SIFT_DETECTOR_H

#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>

extern "C" {
    #include "vl/sift.h"
}


class SiftDetector {

public:
    explicit SiftDetector(double in_peak_threshold = 5,double in_edge_threshold = 5);
    void detect_and_compute(const cv::Mat &img,std::vector<VlSiftKeypoint> &kpts,std::vector<std::vector<float>> &descriptors,bool root_sift = false);
    static void save_sift(std::ofstream &out_file, const std::vector<std::vector<float>> &descriptors);
    static void load_sift(std::ifstream &in_file,std::vector<std::vector<float>> &desceriptors);

private:
    /*
    Extract sift using vlfeat
    parameters:
        vl_sfit, VlSiftFilt*
        data , image pixel data ,to be convert to float
        kpts, keypoint list
        descriptors, descriptor. Need to free the memory after using.
    */
    void vl_sift_extract(VlSiftFilt *vl_sift, vl_sift_pix* data,
                         std::vector<VlSiftKeypoint> &kpts,std::vector<std::vector<float>> &descriptors);

    /*
     * Transfrom to root-sift
     * */
    void convert_sift_to_root_sift(std::vector<std::vector<float>> &descriptors);
private:
    double peak_threshold;
    double edge_threshold;
};

#endif

