
#include "Utility.h"

#include <opencv2/xfeatures2d.hpp>
#include <sys/stat.h>
#include <dirent.h>
#include <vector>

using namespace cv;
using namespace std;

void siftDetecotor::extractFeatures(const std::vector<std::string> &imageFileList,std::vector<std::vector<cv::KeyPoint>> &kptList,std::vector<cv::Mat> &features)
{
    int index = 1;
    int count = 0;
    features.reserve(imageFileList.size());

    auto size = imageFileList.size();
    //size = 20;
    //#pragma omp parallel for
    for(size_t i = 0; i < size; i ++){
        auto str = imageFileList[i];
        Mat des;
        vector<KeyPoint> kpts;
        siftDetecotor::extractFeatures(str,kpts,des);
        features.emplace_back(des); 
        kptList.emplace_back(kpts);
        count += des.rows;
        index ++ ;
    }  
    cout << "Extract #" << index << "# images features done!" << "Count of features:#" << count << endl;
}

void siftDetecotor::rootSift(const cv::Mat &siftFeature,cv::Mat &rootSiftFeature)
{   
    for(int i = 0; i < siftFeature.rows; i ++){
        // Conver to float type
        Mat f;
        siftFeature.row(i).convertTo(f,CV_32FC1);

        normalize(f,f,1,0,NORM_L1); // l1 normalize
        sqrt(f,f); // sqrt-root  root-sift
        rootSiftFeature.push_back(f);
    }
}

void siftDetecotor::rootSift(const std::vector<cv::Mat> &siftFeatures,std::vector<cv::Mat> &rootSiftFeatures)
{   
    auto size = siftFeatures.size();
    //#pragma omp parallel for
    for(size_t i = 0; i < size; i ++){
        Mat root_sift;
        rootSift(siftFeatures[i],root_sift);
        rootSiftFeatures.push_back(root_sift);
    }
}

void siftDetecotor::extractFeatures(const std::string &file,std::vector<cv::KeyPoint> &kpts,cv::Mat &feature)
{
    auto img = imread(file,IMREAD_GRAYSCALE);
    if(img.empty()){
        cerr << "Open image #" << file << " features failed" << endl;
        return;
    }
    cout << "Extract feature from image #" << file << endl;
    auto fdetector = xfeatures2d::SIFT::create(0,3,0.2,10);
    fdetector->detectAndCompute(img,noArray(),kpts,feature);

    if(kpts.size() < 10){
        fdetector = xfeatures2d::SIFT::create();
        fdetector->detectAndCompute(img,noArray(),kpts,feature);
    }
}

void siftDetecotor::extractFeaturesFromImg(const cv::Mat &img,vector<KeyPoint> &kpts,cv::Mat &feature)
{
    auto fdetector = xfeatures2d::SIFT::create(0,3,0.2,10);

    fdetector->detectAndCompute(img,noArray(),kpts,feature);

    if(kpts.size() < 10){
        fdetector = xfeatures2d::SIFT::create();
        fdetector->detectAndCompute(img,noArray(),kpts,feature);
    }
}


void PathManager::get_path_list(const std::string &folder,std::vector<std::string> &file_list)
{
    dirent *file;
    DIR* dir = opendir(folder.c_str());

    std::stringstream ss;
    while((file = readdir(dir)) != nullptr){
        if(std::string(file->d_name) == "." || std::string(file->d_name) == ".."){
            continue;
        }

        ss.str("");
        ss << folder << "/" << file->d_name;
        file_list.emplace_back(ss.str());
    }
}

void PathManager::extractFilename(const std::string &path,std::string &name)
{
    auto dot_pos = path.find_last_of('.');
    if(dot_pos == std::string::npos){
        name = "";
        return ;
    }

    auto pos = path.find_last_of('/');
    if(pos == std::string::npos){
        pos = 0;
        name = path.substr(pos,dot_pos-pos);
        return ;
    }
    name = path.substr(pos+1,dot_pos - pos-1);
}

string PathManager::contact(const std::string filename,std::initializer_list<std::string> folders)
{
    stringstream ss;
    for_each(folders.begin(),folders.end(),[&ss](const string &str){
        ss << str << "/";
    });
    ss << filename;

    return ss.str();
}