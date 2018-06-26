
#include "imageTrainer.h"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/flann/kdtree_index.h>

using namespace std;
using namespace cv;

void ImageTrainer::extract_sift(){
    // 提取图像的sift
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();

    int count = 0;
    for(const string & file: image_file_list){
        cout << "Extracte sift feature #" << file << endl;
        vector<KeyPoint> kpts;
        Mat des;
        Mat img = imread(file);
        CV_Assert(!img.empty());
        sift->detectAndCompute(img,noArray(),kpts,des);
        //Mat norm_des(des.rows,des.cols,CV_32FC1,Scalar::all(0));
        //normalize(des,norm_des,1,0,NormTypes::NORM_L2,CV_32FC1);
        feature_list.push_back(des);
        count ++;
        if(count == 20) break ;
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

void ImageTrainer::vocabulary_bowtrainer(){

    BOWKMeansTrainer bowtrainer(20);

    Mat mat;
    vconcat(feature_list,mat);
	bowtrainer.add(mat);
	
	Mat vocabulary = bowtrainer.cluster();
    cluster_centers = vocabulary;
    FileStorage fs("centers_cv.xml",FileStorage::WRITE);
    fs << "centers" << vocabulary;
    fs.release();
}

void ImageTrainer::vlad_quanlization(const cv::Mat &img,cv::Mat &vlad){

    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    vector<KeyPoint> kpts;
    Mat descriptor;

    sift->detectAndCompute(img,noArray(),kpts,descriptor);

    Mat float_des;
    descriptor.convertTo(float_des,CV_32FC1);

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	BOWImgDescriptorExtractor bowide(matcher);
	bowide.setVocabulary(cluster_centers);

    vector<DMatch> matches;
    matcher->match(float_des,matches);

    Mat quan(k,descriptor.cols,CV_32FC1,Scalar::all(0));

    for( size_t i = 0; i < matches.size(); i++ ){
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx; // cluster index
        CV_Assert( queryIdx == (int)i );
        Mat residual;
        
        subtract(float_des.row(matches[i].queryIdx),cluster_centers.row(matches[i].trainIdx),residual,noArray(), CV_32F);
        add(quan.row(matches[i].trainIdx),residual,quan.row(matches[i].trainIdx),noArray(),quan.type());  
    }

    quan /= norm(quan,NORM_L2);
    quan = quan.reshape(0,1);

    vlad = quan;
}
void ImageTrainer::vlad_quanlization(std::vector<cv::Mat> &vlad){

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	BOWImgDescriptorExtractor bowide(matcher);
	bowide.setVocabulary(cluster_centers);

    for(const Mat &img : feature_list){

        Mat float_des;
        img.convertTo(float_des,CV_32FC1);
        vector<DMatch> matches;
        matcher->match(float_des,matches);

        Mat quan(k,img.cols,CV_32FC1,Scalar::all(0));

        for( size_t i = 0; i < matches.size(); i++ ){
            int queryIdx = matches[i].queryIdx;
            int trainIdx = matches[i].trainIdx; // cluster index
            CV_Assert( queryIdx == (int)i );
            Mat residual;
            
            subtract(float_des.row(matches[i].queryIdx),cluster_centers.row(matches[i].trainIdx),residual,noArray(), CV_32F);
            add(quan.row(matches[i].trainIdx),residual,quan.row(matches[i].trainIdx),noArray(),quan.type());  
        }

        quan /= norm(quan,NORM_L2);
        quan = quan.reshape(0,1);

        vlad.push_back(quan);   
    }

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
    FileStorage fs("des.xml",FileStorage::WRITE);
    fs << "des" << descriptor;
    fs.release();

    // Build kd-tree
    flann::Index kdtree(cluster_centers,flann::KDTreeIndexParams());

    // Search
    vector<float> dist;
    Mat f32_des;
    normalize(descriptor,f32_des,1,0,NORM_L2,CV_32FC1);
    // vlad-encode
    Mat vlad_encode(k,descriptor.cols,CV_32FC1,Scalar::all(0));
    for(int i = 0; i < descriptor.rows; i ++) {
        vector<int> label;
        kdtree.knnSearch(f32_des.row(i),label,dist,3);
        cout << "Quantization:" << label[0] << " Distance:" << dist[0] << endl;

        // vlad-encode
        Mat item;
        subtract(descriptor.row(i),cluster_centers.row(label[0]),item);
        add(vlad_encode.row(label[0]),item,vlad_encode.row(label[0]));
    }

    vlad_encode /= norm(vlad_encode,NORM_L2); // l2 normalization
    vlad = vlad_encode.reshape(0,1);
}

void ImageTrainer::retrieval(const Mat &img,const Mat &vlad_list,std::string &retrieved_image){
    Mat vlad;
    vlad_encode(img,vlad);

    // build kd-tree
    flann::Index kdtree(vlad_list,flann::KDTreeIndexParams());

    //search
    vector<int> result;
    vector<float> dist;
    kdtree.knnSearch(vlad,result,dist,3);

    retrieved_image = image_file_list[result[0]];

    cout << "retrieved image:" << retrieved_image << "distance:" << dist[0] << endl;
}

void ImageTrainer::retireval_bow(const cv::Mat &img,const cv::Mat &vlad_list,std::string &retrieved_image){

    Mat vlad;
    vlad_quanlization(img,vlad);

    // build kd-tree
    flann::Index kdtree(vlad_list,flann::KDTreeIndexParams());

    //search
    vector<int> result;
    vector<float> dist;
    kdtree.knnSearch(vlad,result,dist,3);

    retrieved_image = image_file_list[result[0]];

    cout << "retrieved image:" << retrieved_image << "distance:" << dist[0] << endl;
}