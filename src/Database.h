
#ifndef H__DATABASE__
#define H__DATABASE__

#include "Vocabulary.h"
#include "RootSiftDetector.h"
#include <memory>
#include <string>
#include <vector>

class Item{

public:
    cv::Mat vlad;
    std::vector<cv::Mat> features;
    std::vector<std::string> md5_list;
    std::vector<std::vector<cv::Point2f>> kpts;
    std::string name;
};

class Database{

public:
    Database();
    ~Database();

    /*
        Create database folder
    */
    Database(std::shared_ptr<RootSiftDetector> featureDetector);


    /*
        Set vocabulary
    */
    void setVocabulary(const Vocabulary &voc);

    /*
        Build database from image list
    */
    void buildDatabase(const std::vector<std::string> imageFileList,int pcaDims);


    void save1(const std::string &folder,const std::string &identifier);
    void load1(const std::string &folder,const std::string &identifier);

    bool add(const cv::Mat &img,const std::string &md5,const std::string &groupname);

    void retrieval(const cv::Mat &img,const std::string &groupname,
        std::vector<std::string> &res,std::vector<float> &dists,int k);

    void draw(const cv::Mat &query,const std::vector<std::string> images);

private:

    
    /*
        KNN-match
        Return the count of matched points
    */
    int spatialVerificationRatio(const cv::Mat &des1,const cv::Mat &des2,std::vector<cv::DMatch> &matches);

    /*
        RANSACE refine match
    */
   void spatialVerificationRANSAC(const std::vector<cv::DMatch> &originMatch,   // Origin match,need to refine
                                  const std::vector<cv::KeyPoint> &kpts1,  // keypoints location of query image
                                  const std::vector<cv::KeyPoint> &kpts2,  // keypoints location of train_image
                                  cv::Mat &homography,                          // Homography of the two images, may be usefull future
                                  std::vector<cv::DMatch> &betterMatch);        // Refined matches

private:

    std::shared_ptr<RootSiftDetector> m_featureDetector;
    Vocabulary m_voc;

    std::vector<Item> m_items;

    cv::PCA pca;
};


#endif