
#ifndef __H_SEARCHER
#define __H_SEARCHER

#include <opencv2/opencv.hpp>
#include "Database.h"
#include <memory>
#include <string>
#include <vector>

class Searcher{

public:
    Searcher();
    ~Searcher();

    void init(int keyPointThreshold);
    void setDatabase(std::shared_ptr<Database> db);

    void retrieval(cv::Mat &query,const std::string &group,std::string &md5,double &score);

    void retrieval(std::vector<char> bins,const std::string &group,std::string &md5,double &score);

private:
    int m_keyPointThreshold;

    std::shared_ptr<Database> m_db;
};


#endif