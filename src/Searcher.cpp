
#include "Searcher.h"

using namespace std;
using namespace cv;

Searcher::Searcher(){}
Searcher::~Searcher(){}

void Searcher::init(int keyPointThreshold)
{
    m_keyPointThreshold = keyPointThreshold;
}

void Searcher::setDatabase(std::shared_ptr<Database> db)
{
    m_db = db;
}

void Searcher::retrieval(std::vector<char> bins,const std::string &group,std::string &md5,double &score)
{
    if(bins.empty()){
        cerr << "The buffer of image is empty,return" << endl;
        return ;
    }
    Mat img = imdecode(bins,IMREAD_GRAYSCALE);
    //imwrite("test.jpg",img);
    if(img.empty()){
        cout << "Decode image failed" << endl;
        return ;
    }

    retrieval(img,group,md5,score);
}

void Searcher::retrieval(cv::Mat &query,const std::string &group,std::string &md5,double &score)
{
    Mat left    = query(Rect(0,0,query.cols/2,query.rows));
    Mat right   = query(Rect(query.cols/2,0,query.cols/2,query.rows));

    const int padding   = 50; // pixels
    const int height    = query.rows;
    const int width     = query.cols;

    int k = 50;
    
    auto f = [&group,k,this](Mat &img,string &md5,float &score)->int{
        vector<string> md5List;
        vector<float> scoreList;
        m_db->retrieval(img,group,md5List,scoreList,k);

        md5 = md5List[0];
        score = scoreList[0];

        return scoreList[0];
    };

    // Clip padding
    auto tmp = query(Rect(padding,padding,width-padding,height-padding));
    // Split three blocks(up,center,bootom),retrieval center first
    int tmpHeight   = height - padding;
    int tmpWidth    = width - padding;
    auto up = tmp(Rect(0,0,tmpWidth,tmpHeight / 3));
    auto center = tmp(Rect(0,tmpHeight / 3,tmpWidth,tmpHeight/3));
    auto bottom = tmp(Rect(0,tmpHeight * 2 / 3,tmpWidth,tmpHeight /3 ));

    /*vector<Mat> imgList = {tmp,center,bottom,up};
    for(Mat &mat : imgList){
        if(f(mat) >= m_keyPointThreshold) break;
    }*/

    string allMd5,centerMd5;
    float allScore,centerScore;

    f(tmp,allMd5,allScore);
    f(center,centerMd5,centerScore);

    if(allScore >= m_keyPointThreshold || centerScore >= m_keyPointThreshold){

        md5 = allScore > centerScore ? allMd5 : centerMd5;
        score = allScore > centerScore ? allScore : centerScore;

    }else {
        string upMd5,bottomMd5;
        float upScore,bottomScore;

        f(up,upMd5,upScore);
        f(bottom,bottomMd5,bottomScore);

        if(bottomScore < m_keyPointThreshold && upScore < m_keyPointThreshold){
            md5 = "";
            score = 0;
        }
        else{
            md5 = bottomScore > upScore ? bottomMd5 : upMd5;
            score = bottomScore > upScore ? bottomScore : upScore;
        }
    }
}