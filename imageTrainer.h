
#ifndef IMAGE_TRAINER_H
#define IMAGE_TRAINER_H

#include<vector>
#include<string>
#include<opencv2/opencv.hpp>
#include<opencv2/xfeatures2d.hpp>

class ImageTrainer{

public:
    ImageTrainer(int in_k,const std::vector<std::string> &in_image_file_list):
        k(in_k),image_file_list(in_image_file_list) {}
    
    void bof_encode(std::vector<cv::Mat> &bof);
    void vlad_encode(std::vector<cv::Mat> &vlad);
    void vlad_encode(const cv::Mat &img,cv::Mat &vlad); // vlad encode image
    void extract_sift(); // Extract sift feature
    void vocabulary_kmeans(); // Get visual vocabulary using k-means algorithm  

private:
    const int k; // number of k-means clustering
    std::vector<std::string> image_file_list;
    std::vector<cv::Mat> feature_list; // All feature of images 
    cv::Mat vocabulary; // Visual vocabulary
    cv::Mat cluster_centers; // center of k cluster
    std::vector<int> labels;  // the index of cluster that one feature closest
};

#endif