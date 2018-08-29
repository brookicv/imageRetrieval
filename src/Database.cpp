
#include "Database.h"
#include "Utility.h"

#include <sys/stat.h>
#include <dirent.h>
#include <cstdio>
#include <unistd.h>
#include <algorithm>
#include <memory>

using namespace cv;
using namespace std;

Database::Database(){}
Database::~Database(){}

Database::Database(std::shared_ptr<RootSiftDetector> featureDetector)
{
    m_featureDetector = featureDetector;
}


void Database::setVocabulary(const Vocabulary &voc)
{
    m_voc = voc;
}

void Database::buildDatabase(const std::vector<std::string> imageFileList,int pcaDims)
{
    int count = 1;
    string groupname = "groupyx";

    vector<Mat> vladList;
    for(const string &str : imageFileList){

        Mat img = imread(str,IMREAD_GRAYSCALE);

        //resize(img,img,Size(256,256));
        if(img.empty()) continue;

        //string md5;
        //PathManager::extractFilename(str,md5);
        cout << "Transform " << count << "st image #" << str << endl;
        //add(img,str,"groupyx");
        vector<KeyPoint> kpts;
        Mat f;
        m_featureDetector->detectAndCompute(img,kpts,f);
        Mat vlad;
        m_voc.transform_vlad(f,vlad);

        vladList.emplace_back(vlad);

        auto it = find_if(m_items.begin(),m_items.end(),[&groupname](Item &item){
            return item.name == groupname;
        });

        if(it != m_items.end()){
            it->md5_list.emplace_back(str);
            it->vlad.push_back(vlad);

        } else {
            Item im;
            im.name = groupname;
            im.vlad.push_back(vlad);
            im.md5_list.emplace_back(str);

            m_items.emplace_back(im);
        }
        count ++;
    }

    if(pcaDims > 0){
        Mat tmp;
        vconcat(vladList,tmp);
        pca = PCA(tmp,Mat(),PCA::DATA_AS_ROW,pcaDims);

        for(Item &item : m_items){
            Mat pca_vlad;//(item.vlad.rows,128,item.vlad.type());
            for(int i = 0 ; i < item.vlad.rows; i++){
                Mat m = pca.project(item.vlad.row(i));
                pca_vlad.push_back(m);
            }
            item.vlad = pca_vlad;
        }
    }

}


bool Database::add(const cv::Mat &img,const std::string &md5,const std::string &groupname)
{   
    //Mat tmp;
    //resize_image(img,tmp,480,640);

    vector<KeyPoint> kpts;
    Mat root_sift;

    m_featureDetector->detectAndCompute(img,kpts,root_sift);

    Mat vlad;
    m_voc.transform_vlad(root_sift,vlad);

    auto it = find_if(m_items.begin(),m_items.end(),[&groupname](Item &item){
        return item.name == groupname;
    });

    Mat pca_vlad;
    pca.project(vlad,pca_vlad);

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

void Database::save1(const std::string &folder,const std::string &identifier)
{
    stringstream ss;
    ss << folder << "/db-" << identifier << ".yml";

    FileStorage fs(ss.str(),FileStorage::WRITE);
    
    cout << "-------------------------------------------" << endl;
    cout << "create database file:" << ss.str() << endl;

    if(fs.isOpened()){

        fs << "groups" << "[";
        for(const Item &item : m_items){
            fs << "{" << "name" << item.name ;
            fs << "list" << "[";

            for(int i = 0; i < item.md5_list.size(); i ++){
                fs << "{" << "md5" << item.md5_list[i] << "vlad" << item.vlad.row(i) << "}";
            }
            fs << "]";
            fs << "}";
        }
        fs << "]";
    }
    fs.release();

    ss.str("");
    ss << folder << "/pca-" << identifier << ".yml";
    FileStorage pca_fs(ss.str(),FileStorage::WRITE);
    pca.write(pca_fs);
    pca_fs.release();
}

void Database::load1(const std::string &folder,const std::string &identifier)
{
    stringstream ss;
    ss << folder << "/db-" << identifier << ".yml";

    FileStorage fs(ss.str(),FileStorage::READ);

    if(fs.isOpened()){

        FileNode groupNode = fs["groups"];

        for(auto it = groupNode.begin(); it != groupNode.end(); it ++) {
            Item item;
            (*it)["name"] >> item.name;

            FileNode list = (*it)["list"];
            for(auto list_it = list.begin(); list_it != list.end(); list_it ++){
                Mat vlad;
                string md5;

                (*list_it)["md5"] >> md5;
                (*list_it)["vlad"] >> vlad;

                item.md5_list.emplace_back(md5);
                item.vlad.push_back(vlad);
            }
            m_items.push_back(item);
        }
    }
    fs.release();

    ss.str("");
    ss << folder << "/pca-" << identifier << ".yml";
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

    vector<KeyPoint> kpts;
    Mat root_sift;
    m_featureDetector->detectAndCompute(img,kpts,root_sift);
    
    Mat vlad;
    m_voc.transform_vlad(root_sift,vlad);
    
    flann::KDTreeIndexParams params(1);
    flann::Index retrieval_index;

    retrieval_index.build(it->vlad,params);

    // pca project
    Mat pca_vlad ;

    if(!pca.eigenvalues.empty() && !pca.eigenvectors.empty()){
        pca.project(vlad,pca_vlad);
    }else{
        pca_vlad = vlad;
    }

    vector<int> indexs;
    retrieval_index.knnSearch(pca_vlad,indexs,dists,k,flann::SearchParams(-1));

    // Spatial Verfication
    struct RefineMatch{
        int idx;
        Mat homography;
        vector<DMatch> betterMatch;
        int count;
    };

    vector<RefineMatch> refineMatchList;

    for(int i = 0 ; i < k ; i ++){

        vector<KeyPoint> t_kpts;
        Mat t_root_rift;
        Mat tmp = imread(it->md5_list[indexs[i]]);
        
        m_featureDetector->detectAndCompute(tmp,t_kpts,t_root_rift);
       
        vector<DMatch> betterMatches;
        auto count = spatialVerificationRatio(root_sift,t_root_rift,betterMatches);    
        if(count > 0){
            RefineMatch rm ;
            rm.idx = indexs[i];
            rm.count = count;
            rm.betterMatch = betterMatches;
            refineMatchList.emplace_back(rm);
        }
    }

    sort(refineMatchList.begin(),refineMatchList.end(),[](const RefineMatch &rm1,const RefineMatch &rm2){
        return rm1.count > rm2.count;
    });

    for(const RefineMatch &rm : refineMatchList){
        cout << "index:" << rm.idx << ",count:" << rm.count << ",name:" << it->md5_list[rm.idx] << endl;
    }

    int top = 10;
    int count = top;
    if(count > refineMatchList.size()) count = refineMatchList.size();
    
    cout << "-----------origin result----------------" << endl;
    for(int i = 0; i < count; i ++){
        cout << it->md5_list[indexs[i]] << endl;
        cout << dists[i] << endl;
    }

    dists.clear();
    cout << "-----------Spatial Verfication result----------------" << endl;
    for(int i = 0; i < count; i ++){
        cout << it->md5_list[refineMatchList[i].idx] << endl;
        cout << refineMatchList[i].count << endl;

        res.push_back(it->md5_list[refineMatchList[i].idx]);
        dists.push_back(refineMatchList[i].count);
    }

    // Average query expansion
    /*Mat sum(1,it->vlad.cols,CV_32FC1,Scalar::all(0.0));
    for(int i = 0; i < count; i ++){
        sum += it->vlad.row(refineMatchList[i].idx);
    }

    sum += pca_vlad;

    sum /= (count + 1);
    dists.clear();
    indexs.clear();
    retrieval_index.knnSearch(sum,indexs,dists,10);

    for(int i = 0; i < 10;  i ++){
        res.push_back(it->md5_list[indexs[i]]);
    }*/
}

int Database::spatialVerificationRatio(const cv::Mat &des1,const cv::Mat &des2,vector<DMatch> &matches)
{
    const float minRatio = 1.f / 1.3f;
    const int k = 2;

    //Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    auto matcher = DescriptorMatcher::create("BruteForce");
    vector<vector<DMatch>> knnMatches;
    matcher->knnMatch(des1, des2, knnMatches, k);

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
                                const std::vector<cv::KeyPoint> &kpts1,           // keypoints location of query image
                                const std::vector<cv::KeyPoint> &kpts2,           // keypoints location of train_image
                                cv::Mat &homography,                                   // Homography of the two images, may be usefull future
                                std::vector<cv::DMatch> &betterMatch)                  // Refined matches
{
    // Assignment the points
    vector<Point2f> queryPoints;
    vector<Point2f> trainPoints;

    for(const DMatch & match : originMatch){
        queryPoints.emplace_back(kpts1[match.queryIdx].pt);
        trainPoints.emplace_back(kpts2[match.trainIdx].pt);
    }
    // Find homography matrix and get inliers mask      
    std::vector<unsigned char> inliersMask(originMatch.size());
    float reprojectionThreshold = 0.01f;     
    homography = cv::findHomography(queryPoints,trainPoints,CV_FM_RANSAC,reprojectionThreshold,inliersMask);      
        
    for (size_t i=0; i<inliersMask.size(); i++)      
    {      
        if (inliersMask[i])      
            betterMatch.push_back(originMatch[i]);      
    }
}

void Database::draw(const cv::Mat &query,const std::vector<std::string> images)
{
    for(int i = 0; i < images.size(); i ++){
        auto mat = imread(images[i]);

        Mat rf;
        vector<KeyPoint> kpts;
        m_featureDetector->detectAndCompute(mat,kpts,rf);

        vector<KeyPoint> kpts1;
        Mat rf1;
        m_featureDetector->detectAndCompute(query,kpts1,rf1);
        
        //Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
        auto matcher = DescriptorMatcher::create("BruteForce");
        vector<DMatch> matches;
        matcher->match(rf1,rf,matches);

        Mat tmp;
        drawMatches(query,kpts1,mat,kpts,matches,tmp);
        imshow("res",tmp);
    }
}