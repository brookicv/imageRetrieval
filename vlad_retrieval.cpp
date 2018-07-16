
#include "utils.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "imageTrainer.h"
#include <opencv2/flann/kdtree_index.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;

void vocabulary_training(const vector<string> &image_file_list,int k,Mat &vocabulary,vector<Mat> &features){
    // 提取图像的sift
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(0,3,0.16,10);
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

void training(const string &image_folder,const string &data_folder,int k) {

    vector<string> image_file_list;
    get_file_name_list(image_folder,image_file_list);

    TickMeter tm;
    Mat vocabulary;
    vector<Mat> features;
    stringstream ss;
    ss << data_folder << "/" << "vocabulary.xml";
    tm.start();
    vocabulary_training(image_file_list,k,vocabulary,features);
    tm.stop();
    cout << "k-means to get cluster costs time:" << tm.getTimeSec() << "s." << endl;

    FileStorage fs_voc(ss.str(),FileStorage::WRITE);
    fs_voc << "vocabulary" << vocabulary;
    fs_voc.release();

    vector<Mat> vlad_list;
    vlad_quantization(features,vocabulary,vlad_list);
    ss.str("");
    ss << data_folder << "/" << "vlad.xml";
    FileStorage fs_vlad(ss.str(),FileStorage::WRITE);
    Mat vlad;
    vconcat(vlad_list,vlad);
    fs_vlad << "vlad" << vlad;
    fs_vlad.release();

    ss.str("");
    ss << data_folder << "/" << "search.index";
    flann::Index retrieval_index;
    build_index(vlad_list,retrieval_index,ss.str());
}

int main()
{
    //const string image_folder = "/home/brook/git_folder/image_retrieval/images";
    const string image_folder = "/home/test/git/jpg";
    const string data_folder = "../data";

    int k  = 1000;
    training(image_folder,data_folder,k);

    //Mat img = imread("/home/brook/git_folder/image_retrieval/test.png");
    //int index;
    //double dist = retrieval_vlad(img,vocabulary,retrieval_index,index);
    return 0;
}  