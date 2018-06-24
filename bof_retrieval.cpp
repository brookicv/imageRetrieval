
#include "utils.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace std;
using namespace cv;

void bof_train(const string &image_folder,int k,vector<Mat> &bof) {

    vector<string> image_file_list;
    get_file_name_list(image_folder,image_file_list);

    // 提取图像的sift
    vector<Mat> descriptor_list;
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    for(const string & file: image_file_list){
        cout << "Extracte sift feature #" << file << endl;
        vector<KeyPoint> kpts;
        Mat des;
        Mat img = imread(file);
        CV_Assert(!img.empty());
        sift->detectAndCompute(img,noArray(),kpts,des);
        descriptor_list.push_back(des);
    }

    // 将各个图像的sift特征组合到一起
    Mat descriptor_stack;
    vconcat(descriptor_list,descriptor_stack);

    // 聚类
    Mat cluster_centers;
    vector<int> labels;
    kmeans(descriptor_stack,k,labels,TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 
        10, 1.0),3, KMEANS_RANDOM_CENTERS,cluster_centers);

    // labels已经得到了每个样本（特征点）所属的簇，需要进行统计得到每一张图像的BoF
    int index = 0;
    for(Mat img : descriptor_list){
        // For all keypoints of each image 
        auto cluster = new int[k];
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

int main()
{
    const string image_folder = "/home/book/git/imageRetrieval/image";

    vector<Mat> bof;
    bof_train(image_folder,20,bof);

    return 0;
}