
#include "utils.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "imageTrainer.h"
#include <opencv2/flann/kdtree_index.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

void kd_test(){

    // Build kd tree
    Point2f p1(2,3),   p2(5,4),p3(9, 6),  p4(4,7),  p5(8,1),p6(7, 2);
    vector<cv::Point2f> source;
    source.push_back(p1);
    source.push_back(p2);
    source.push_back(p3);
    source.push_back(p4);
    source.push_back(p5);
    source.push_back(p6);
    source.push_back(Point2f(1,1.5));
    flann::KDTreeIndexParams indexParams(2);
    flann::Index kdtree(cv::Mat(source).reshape(1), indexParams);

    // Query
    vector<float> query;
    Point2f pt(0.1,1);
    query.push_back(pt.x);
    query.push_back(pt.y);
    int k = 4; //number of nearest neighbors
    vector<int> indices(k);//找到点的索引
    vector<float> dists(k);

    flann::SearchParams params(128);
    kdtree.knnSearch(query, indices, dists, k,params);
    cout<<indices[0]<< " : dist->" << dists[0] << " point:" << source[indices[0]] << endl;
    cout<<indices[1]<<" : dist->" << dists[1] << " point:" << source[indices[1]] << endl;
    cout<<indices[2]<<" : dist->" << dists[2] << " point:" << source[indices[2]] << endl;
    cout<<indices[3]<<" : dist->" << dists[3] << " point:" << source[indices[3]] << endl;
}

void vocabulary_training(const vector<string> &image_file_list,int k,Mat &vocabulary,vector<Mat> &features){
    // 提取图像的sift
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(0,3,0.1,10);
    int index = 1;
    for(const string & file: image_file_list){
        cout << "Extract sift feature " << index << "th image" << "#" <<  file << endl;
        vector<KeyPoint> kpts;
        Mat des;
        Mat img = imread(file);
        CV_Assert(!img.empty());
        sift->detectAndCompute(img,noArray(),kpts,des);
        //Mat norm_des(des.rows,des.cols,CV_32FC1,Scalar::all(0));
        //normalize(des,norm_des,1,0,NormTypes::NORM_L2,CV_32FC1);
        features.push_back(des);

        index ++ ;

        if(index >= 20) break ;
    }

    cout << "Extract #" << index << "# images sift feature" << endl;
    cout << "Get visual vocabulary using k-means algorithm..." << endl;

    Mat feature_list;
    vconcat(features,feature_list);

    BOWKMeansTrainer bow_trainer(k);
    bow_trainer.add(feature_list);

    vocabulary =  bow_trainer.cluster();

    cout << "Generate visual vocabulary is complete" << endl;
}

void vlad_quantization(const Mat &feature,const Mat &vocabulary,Mat &vlad){
    Mat float_des;
    feature.convertTo(float_des,CV_32FC1);

    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    vector<DMatch> matches;
    matcher->match(float_des,vocabulary,matches);

    Mat quan(vocabulary.rows,feature.cols,CV_32FC1,Scalar::all(0));

    for( size_t i = 0; i < matches.size(); i++ ){
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx; // cluster index
        CV_Assert( queryIdx == (int)i );
        Mat residual;
        
        subtract(float_des.row(queryIdx),vocabulary.row(trainIdx),residual,noArray(), CV_32F);
        add(quan.row(trainIdx),residual,quan.row(trainIdx),noArray(),quan.type());  
    }
    quan /= norm(quan,NORM_L2);
    vlad = quan.reshape(0,1);
}

void vlad_quantization(const vector<Mat> &feature_list,const Mat &vocabulary,vector<Mat> &vlad_list){

    for(const Mat &f : feature_list){
        Mat vlad;
        vlad_quantization(f,vocabulary,vlad);
        vlad_list.push_back(vlad);
    }
}

void iamge_vlad_quantization(const Mat &img,const Mat &vocabulary,Mat &vlad){

    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();

    vector<KeyPoint> kpts;
    Mat descriptor;
    sift->detectAndCompute(img,noArray(),kpts,descriptor);

    vlad_quantization(descriptor,vocabulary,vlad);
}

void build_index(const vector<Mat> &vlad,flann::Index &retrieval_index,const string &index_file_name = "") {

    Mat tmp;
    vconcat(vlad,tmp);

    cout << "build search index..." << endl;
    flann::KDTreeIndexParams params;
    retrieval_index.build(tmp,params);
    
    if(!index_file_name.empty()){
        retrieval_index.save(index_file_name);
    }
}

double retrieval_vlad(const Mat &img,const Mat &vocabulary,flann::Index &retrieval_index,int &index) {
    Mat vlad;
    iamge_vlad_quantization(img,vocabulary,vlad);

    vector<int> indexs;
    vector<float> distances;

    int k = 3;
    retrieval_index.knnSearch(vlad,indexs,distances,k);

    index = indexs[0];

    return distances[0];
}

int main()
{
    const string image_folder = "/home/brook/git_folder/image_retrieval/images";
    //const string image_folder = "/home/book/git/imageRetrieval/images";
    vector<string> image_file_list;
    get_file_name_list(image_folder,image_file_list);

    Mat vocabulary;
    vector<Mat> features;
    vocabulary_training(image_file_list,10,vocabulary,features);

    vector<Mat> vlad_list;
    vlad_quantization(features,vocabulary,vlad_list);

    flann::Index retrieval_index;
    build_index(vlad_list,retrieval_index);

    Mat img = imread("/home/brook/git_folder/image_retrieval/test.png");
    int index;
    double dist = retrieval_vlad(img,vocabulary,retrieval_index,index);
    return 0;
}  