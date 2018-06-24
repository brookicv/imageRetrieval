
#ifndef VL_SIFT_DETECTOR_H
#define VL_SIFT_DETECTOR_H

#include <vector>
#include <cmath>

extern "C" {
    #include "vl/sift.h"
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
                    std::vector<VlSiftKeypoint> &kpts,std::vector<float*> &descriptors) {
    
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

void convert_sift_to_root_sift(std::vector<float*> &descriptors){
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

void vl_root_sift_extract(VlSiftFilt *vl_sift, vl_sift_pix* data,std::vector<VlSiftKeypoint> &kpts,std::vector<float*> &descriptors) {

    vl_sift_extract(vl_sift,data,kpts,descriptors);
    convert_sift_to_root_sift(descriptors);
}

#endif

