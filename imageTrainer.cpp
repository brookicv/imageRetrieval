
#include "imageTrainer.h"
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

void ImageTrainer::extract_sift(){
    // 提取图像的sift
    int count = 0;
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    for(const string & file: image_file_list){
        cout << "Extracte sift feature #" << file << endl;
        vector<KeyPoint> kpts;
        Mat des;
        Mat img = imread(file);
        CV_Assert(!img.empty());
        sift->detectAndCompute(img,noArray(),kpts,des);
        feature_list.push_back(des);
        count ++ ;
        if(count > 20) {
            break;
        }
    }
}

void ImageTrainer::vocabulary_kmeans(){
    // 将各个图像的sift特征组合到一起
    Mat descriptor_stack;
    vconcat(feature_list,descriptor_stack);

    // Cluster to get vocabulary
    kmeans(descriptor_stack,k,labels,TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 
        10, 1.0),3, KMEANS_RANDOM_CENTERS,cluster_centers);
}

void ImageTrainer::bof_encode(vector<Mat> &bof){
    // Bof Encode
    // labels已经得到了每个样本（特征点）所属的簇，需要进行统计得到每一张图像的BoF
    int index = 0;
    for(const Mat& img : feature_list){
        // For all keypoints of each image 
        auto cluster = new int[k];
        // Compute TF
        for(int i = 0; i < img.rows; i ++){
            cluster[labels[index]] ++;
            index ++;
        }

        Mat mat(1,k,CV_32S);
        auto ptr = mat.ptr<int>(0);
        mempcpy(ptr,cluster,sizeof(int) * k);

        bof.push_back(mat);
        delete cluster;
    }
}

void ImageTrainer::vlad_encode(vector<Mat> &vlad){
    // VLAD Encode
    // 同一个聚类中心的残差的和
    int index = 0;
    for(const Mat &img : feature_list){
        Mat item(k,img.cols,CV_32FC1,Scalar::all(0));
        for(int i = 0; i < img.rows; i ++) {
            auto label = labels[index]; // cluster of the  index th feature
            Mat center  = cluster_centers.row(label); // center of the cluster
            Mat des     = img.row(i);
            subtract(center,des,des);
            add(item.row(label),des,item.row(label)); 
        }

        item /= norm(item,NORM_L2); // l2 normalize
        item = item.reshape(0,1); // k * d vector represent for one image
        vlad.push_back(item);
    }
}

void ImageTrainer::vlad_encode(const Mat &img,Mat &vlad){

    // Extract sift feature
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    vector<KeyPoint> kpts;
    Mat descriptor;

    sift->detectAndCompute(img,noArray(),kpts,descriptor);

    

}