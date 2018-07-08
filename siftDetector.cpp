
#include "siftDetector.h"

using namespace std;
using namespace cv;

SiftDetector::SiftDetector(double in_peak_threshold,double in_edge_threshold): peak_threshold(in_peak_threshold),edge_threshold(in_edge_threshold){}

void SiftDetector::detect_and_compute(const cv::Mat &img, std::vector<VlSiftKeypoint> &kpts,
                                      std::vector<std::vector<float>> &descriptors, bool root_sift) {
    // Only accept gray image
    assert(img.channels() == 1);

    VlSiftFilt *vl_sift = vl_sift_new(img.cols,img.rows,0,3,0);
    vl_sift_set_edge_thresh(vl_sift,edge_threshold);
    vl_sift_set_peak_thresh(vl_sift,peak_threshold);


    Mat float_img;
    img.convertTo(float_img,CV_32F);

    auto data = (vl_sift_pix*)float_img.data;
    vl_sift_extract(vl_sift,data,kpts,descriptors);
    if(root_sift){
        convert_sift_to_root_sift(descriptors);
    }
}

void SiftDetector::vl_sift_extract(VlSiftFilt *vl_sift, vl_sift_pix *data, std::vector<VlSiftKeypoint> &kpts,
                                   std::vector<std::vector<float>> &descriptors) {
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
                    //float *des = new float[128];
                    vector<float> des(128);
                    vl_sift_calc_keypoint_descriptor(vl_sift,des.data(),pKpts,angles[0]);
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

void SiftDetector::convert_sift_to_root_sift(std::vector<std::vector<float>> &descriptors) {
    /*
        Trans sift to rootSift
        1. L1 normalize each descriptor(128-dims vector)
        2. sqrt-root each element
        3. [option] L2-normalize descriptor vector
    */
    for(vector<float> &vec : descriptors) {
        float sum = 0;
        auto ptr = vec.data();
        for(int i = 0; i < 128; i ++)
            sum += *(ptr + i);
        for(int i = 0; i < 128; i ++){
            *(ptr + i) /= sum;
            *(ptr + i) = sqrt(*(ptr + i));
        }
    }
}

void SiftDetector::save_sift(std::ofstream &out_file, const std::vector<std::vector<float>> &descriptors) {

}

void SiftDetector::load_sift(std::ifstream &in_file, std::vector<std::vector<float>> &desceriptors) {

}
