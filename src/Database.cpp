
#include "Database.h"
#include "Utility.h"

#include <sys/stat.h>
#include <dirent.h>
#include <cstdio>
#include <unistd.h>
#include <algorithm>

using namespace cv;
using namespace std;

Database::Database(){}
Database::~Database(){}

Database::Database(const std::string &name,const std::string &root):m_name(name),m_root(root)
{
    DIR *dir = opendir(m_root.c_str());

    // Don't need to create the database folder
    if(dir != nullptr){
        closedir(dir);
        return ;
    }
    
    // Create database folder
    mkdir(m_root.c_str(),S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
}

bool Database::check() const 
{
    DIR *dir = opendir(m_root.c_str());

    // The folder is exist
    if(dir != nullptr){
        closedir(dir);
        return true;
    }

    return false;
}

void Database::setVocabulary(const Vocabulary &voc)
{
    m_voc = voc;
}

void Database::buildDatabase(const std::vector<std::string> imageFileList)
{
    for(const string &str : imageFileList){

        Mat img = imread(str,IMREAD_GRAYSCALE);
        if(img.empty()) continue;

        string md5;
        PathManager::extractFilename(str,md5);
        add(img,md5,"groupyx");
    }
}

void Database::pcaOperation()
{
    vector<Mat> vlads;
    for(const Item &i : m_items){
        vlads.push_back(i.vlad);
    }

    Mat tmp;
    vconcat(vlads,tmp);

    pca = PCA(tmp,Mat(),PCA::DATA_AS_ROW,128);

    for(Item &item : m_items){
        Mat pca_vlad;//(item.vlad.rows,128,item.vlad.type());
        for(int i = 0 ; i < item.vlad.rows; i++){
            Mat m = pca.project(item.vlad.row(i));
            pca_vlad.push_back(m);
        }
        item.vlad = pca_vlad;
    }
}

bool Database::add(const cv::Mat &img,const std::string &md5,const std::string &groupname)
{

    Mat feature;
    vector<KeyPoint> kpts;
    siftDetecotor::extractFeaturesFromImg(img,kpts,feature);

    Mat root_sift;
    siftDetecotor::rootSift(feature,root_sift);

    Mat vlad;
    m_voc.transform_vlad(root_sift,vlad);

    auto it = find_if(m_items.begin(),m_items.end(),[&groupname](Item &item){
        return item.name == groupname;
    });

    

    if(it != m_items.end()){
        it->md5_list.emplace_back(md5);
        it->vlad.push_back(vlad);
        it->features.push_back(root_sift);

        // Push keypoints location
        vector<Point2f> locations;
        for(const KeyPoint &kpt : kpts){
            locations.emplace_back(kpt.pt);
        }
        it->kpts.emplace_back(locations);

        return true;
    }

    Item im;
    im.name = groupname;
    im.vlad.push_back(vlad);
    im.md5_list.emplace_back(md5);
    im.features.push_back(root_sift);
    // Push keypoints location
    vector<Point2f> locations;
    for(const KeyPoint &kpt : kpts){
        locations.emplace_back(kpt.pt);
    }
    im.kpts.emplace_back(locations);

    m_items.emplace_back(im);
    return true;
}

void Database::save()
{
    stringstream ss;
    ss << m_root << "/" << m_name << ".yml";

    FileStorage fs(ss.str(),FileStorage::WRITE);


    if(fs.isOpened()){

        fs << "groups" << "[";
        for(const Item &item : m_items){
            fs << "{" << "name" << item.name ;
            fs << "list" << "[";

            for(int i = 0; i < item.md5_list.size(); i ++){
                fs << "{" << "md5" << item.md5_list[i] << "vlad" << item.vlad.row(i) 
                << "feature" << item.features[i] << "points" << item.kpts[i] << "}";
            }

            fs << "]";
            fs << "}";
        }
        fs << "]";
    }
    fs.release();

    ss.str("");
    ss << m_root << "/" << m_name << "pca.yml";
    FileStorage pca_fs(ss.str(),FileStorage::WRITE);
    pca.write(pca_fs);
    pca_fs.release();
}

void Database::load()
{
    stringstream ss;
    ss << m_root << "/" << m_name << ".yml";

    FileStorage fs(ss.str(),FileStorage::READ);

    if(fs.isOpened()){

        FileNode groupNode = fs["groups"];

        for(auto it = groupNode.begin(); it != groupNode.end(); it ++) {
            Item item;
            (*it)["name"] >> item.name;

            FileNode list = (*it)["list"];
            for(auto list_it = list.begin(); list_it != list.end(); list_it ++){
                Mat vlad,feature;
                string md5;
                vector<Point2f> kpts;

                (*list_it)["md5"] >> md5;
                (*list_it)["vlad"] >> vlad;
                (*list_it)["feature"] >> feature;
                (*list_it)["points"] >> kpts;

                item.md5_list.emplace_back(md5);
                item.vlad.push_back(vlad);
                item.features.push_back(feature);
                item.kpts.push_back(kpts);
            }
            m_items.push_back(item);
        }
    }
    fs.release();

    ss.str("");
    ss << m_root << "/" << m_name << "pca.yml";
    FileStorage pca_fs(ss.str(),FileStorage::READ);
    if(pca_fs.isOpened()){
        pca.read(pca_fs.root());
        pca_fs.release();
    }
}




void Database::retrieval(const cv::Mat &img,const std::string &groupname,std::vector<std::string> &res,std::vector<float> &dists,int k)
{
    auto it = find_if(m_items.begin(),m_items.end(),[&groupname](const Item item){
        return item.name == groupname;
    });

    Mat feature;
    vector<KeyPoint> kpts;
    siftDetecotor::extractFeaturesFromImg(img,kpts,feature);

    Mat root_sift;
    siftDetecotor::rootSift(feature,root_sift);

    Mat vlad;
    m_voc.transform_vlad(root_sift,vlad);

    flann::KDTreeIndexParams params;
    flann::Index retrieval_index;

    flann::LshIndexParams lsh_params(6,8,2);
    retrieval_index.build(it->vlad,params);

    // pca project
    Mat pca_vlad;
    pca.project(vlad,pca_vlad);

    vector<int> indexs;
    retrieval_index.knnSearch(pca_vlad,indexs,dists,k);

    // Spatial Verfication
    int match_threshold = 1;
    int top = 10;
    
    //vector<int> good_index;

    struct RefineMatch{
        int idx;
        Mat homography;
        vector<DMatch> betterMatch;
        int count;
    };

    vector<RefineMatch> refineMatchList;

    for(int i = 0 ; i < k ; i ++){    
        auto count = spatialVerificationRatio(root_sift,it->features[indexs[i]]);    
        if(count >= match_threshold){
            RefineMatch rm ;
            rm.idx = indexs[i];
            rm.count = count;

            refineMatchList.emplace_back(rm);
        }
    }

    sort(refineMatchList.begin(),refineMatchList.end(),[](const RefineMatch &rm1,const RefineMatch &rm2){
        return rm1.count > rm2.count;
    });

    for(const RefineMatch &rm : refineMatchList){
        cout << "index:" << rm.idx << ",count:" << rm.count << ",name:" << it->md5_list[rm.idx] << endl;
    }

    int count = top;
    if(count > refineMatchList.size()) count = refineMatchList.size();

    for(int i = 0; i < count; i ++){
        res.push_back(it->md5_list[refineMatchList[i].idx]);
    }
    /*
    // Average query expansion
    int count = top;
    if(count > refineMatchList.size()) count = refineMatchList.size();

    
    Mat sum(1,it->vlad.cols,CV_32FC1,Scalar::all(0.0));
    for(int i = 0; i < count; i ++){
        sum += it->vlad.row(refineMatchList[i].idx);
    }

    sum += pca_vlad;

    sum /= (top + 1);

    retrieval_index.knnSearch(sum,indexs,dists,10);

    for(int i = 0; i < 10;  i ++){
        res.push_back(it->md5_list[indexs[i]]);
    }*/
}

int Database::spatialVerificationRatio(const cv::Mat &des1,const cv::Mat &des2)
{
    const float minRatio = 1.f / 1.3f;
    const int k = 2;

    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    vector<vector<DMatch>> knnMatches;
    matcher->knnMatch(des1, des2, knnMatches, k);

    vector<DMatch> matches;
    for (size_t i = 0; i < knnMatches.size(); i++) {
        const DMatch& bestMatch = knnMatches[i][0];
        const DMatch& betterMatch = knnMatches[i][1];

        float  distanceRatio = bestMatch.distance / betterMatch.distance;
        if (distanceRatio < minRatio)
            matches.push_back(bestMatch);
    }

    return matches.size();
}


void Database::spatialVerificationRANSAC(const std::vector<cv::DMatch> &originMatch,   // Origin match,need to refine
                                const std::vector<cv::Point2f> &points1,           // keypoints location of query image
                                const std::vector<cv::Point2f> &points2,           // keypoints location of train_image
                                cv::Mat &homography,                                   // Homography of the two images, may be usefull future
                                std::vector<cv::DMatch> &betterMatch)                  // Refined matches
{
    // Assignment the points
    vector<Point2f> queryPoints;
    vector<Point2f> trainPoints;

    for(const DMatch & match : originMatch){
        queryPoints.emplace_back(points1[match.queryIdx]);
        trainPoints.emplace_back(points2[match.trainIdx]);
    }
    // Find homography matrix and get inliers mask      
    std::vector<unsigned char> inliersMask(originMatch.size());
    float reprojectionThreshold = 0.1f;     
    homography = cv::findHomography(queryPoints,trainPoints,CV_FM_RANSAC,reprojectionThreshold,inliersMask);      
        
    for (size_t i=0; i<inliersMask.size(); i++)      
    {      
        if (inliersMask[i])      
            betterMatch.push_back(originMatch[i]);      
    }
}