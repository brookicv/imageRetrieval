/**
 * File: Vocabulary.h
 * Date: 2018/8/6
 * Author: LiQiang
 * Description: Generate vocabulary from the set of image features, 
 *              and transform a image to bow/vlad 
 *
 */

#ifndef H__VOCABULARY__
#define H__VOCABULARY__

#include "RootSiftDetector.h"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <memory>

class Vocabulary{

public:
    Vocabulary();
    ~Vocabulary();

    Vocabulary(std::shared_ptr<RootSiftDetector> featureDetector,int k);
public:
    /*
        Create from set of features
    */
    void create(const std::vector<cv::Mat> &features);

    /*
        Create from list of image
    */
    void create(const std::vector<std::string> &imageFileList);

    /*
        Load vocabulary from the file
    */
    bool load(const std::string &filename);

    void setFeatureDetector(std::shared_ptr<RootSiftDetector> featureDetector);

    /*
        Save vocabulary to the local file
    */
    bool save(const std::string &filename);

    /*
        Transform bow from the feature of image
    */
    void transform_bow(const cv::Mat &f,std::vector<int> &bow);

    /*
        Transform vlad from the feature of image
    */
    void transform_vlad(const cv::Mat &f,cv::Mat &vlad);

    /*
        Return vocabulary
    */
    void getVocabulary(cv::Mat &voc) const;

    /*
        Set vocabulary
    */
   void setVocabulary(cv::Mat &voc);
    
private:
    cv::Mat m_voc;  //  Vocabulary
    int m_k;        //  Size of the vocabulary

    std::shared_ptr<RootSiftDetector> m_featureDetector;
};

#endif