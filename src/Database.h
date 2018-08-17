
#ifndef H__DATABASE__
#define H__DATABASE__

#include "Vocabulary.h"
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
    Database(const std::string &name,const std::string &root);

    bool check() const;

    /*
        Set vocabulary
    */
    void setVocabulary(const Vocabulary &voc);

    /*
        Build database from image list
    */
    void buildDatabase(const std::vector<std::string> imageFileList);

    void pcaOperation();

    void save();
    void load();

    bool add(const cv::Mat &img,const std::string &md5,const std::string &groupname);

    void retrieval(const cv::Mat &img,const std::string &groupname,
        std::vector<std::string> &res,std::vector<float> &dists,int k);

private:
    /*
        KNN-match
        Return the count of matched points
    */
    int spatialVerificationRatio(const cv::Mat &des1,const cv::Mat &des2);

    /*
        RANSACE refine match
    */
   void spatialVerificationRANSAC(const std::vector<cv::DMatch> &originMatch,   // Origin match,need to refine
                                  const std::vector<cv::Point2f> &queryPoints,  // keypoints location of query image
                                  const std::vector<cv::Point2f> &trainPoints,  // keypoints location of train_image
                                  cv::Mat &homography,                          // Homography of the two images, may be usefull future
                                  std::vector<cv::DMatch> &betterMatch);        // Refined matches

private:
    std::string m_name;
    std::string m_root;

    Vocabulary m_voc;

    std::vector<Item> m_items;

    cv::PCA pca;
};


#endif