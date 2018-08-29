
#include "Vocabulary.h"
#include "Utility.h"

using namespace cv;
using namespace std;

Vocabulary::Vocabulary(){}
Vocabulary::~Vocabulary(){}

Vocabulary::Vocabulary(std::shared_ptr<RootSiftDetector> featureDetector,int k)
{
    m_featureDetector   = featureDetector;
    m_k                 = k;
}   

void Vocabulary::create(const std::vector<cv::Mat> &features)
{
    Mat f;
    vconcat(features,f);
    vector<int> labels;
    kmeans(f,m_k,labels,TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,100,0.01),3,cv::KMEANS_PP_CENTERS,m_voc);
}

void Vocabulary::create(const std::vector<std::string> &imageFileList)
{
    vector<Mat> features;

    int count = 1;
    int sum = 0;
    for(const string &str : imageFileList){

        cout << "Extracte feature from #" << count << "st images # " << str << endl;
        Mat img = imread(str);
        vector<KeyPoint> kpts;
        Mat f;
        m_featureDetector->detectAndCompute(img,kpts,f);
        //m_featureDetector->vlfeatDetectAndCompute(img,f);
        cout << "### Features count:" << f.rows << endl;

        features.emplace_back(f);

        sum += f.rows;
        count ++;
    }

    cout << "Extracte features: " << sum << " from " << count << " images" << endl;

    create(features);
}

void Vocabulary::transform_bow(const cv::Mat &f,std::vector<int> &bow)
{
    // Find the nearest center
    //Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();

    auto matcher = DescriptorMatcher::create("BruteForce");
    vector<DMatch> matches;
    matcher->match(f,m_voc,matches);

    bow = vector<int>(m_k,0);
    // Frequency
    // trainIdx => center index 
    for_each(matches.begin(),matches.end(),[&bow](const DMatch &match){
        bow[match.trainIdx] ++; // Compute word frequency
    });
}

void Vocabulary::transform_vlad(const cv::Mat &f,cv::Mat &vlad)
{
    // Find the nearest center
    //Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    auto matcher = DescriptorMatcher::create("BruteForce");
    vector<DMatch> matches;
    matcher->match(f,m_voc,matches);


    // Compute vlad
    Mat responseHist(m_voc.rows,f.cols,CV_32FC1,Scalar::all(0));
    for( size_t i = 0; i < matches.size(); i++ ){
        auto queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx; // cluster index
        Mat residual;
        subtract(f.row(queryIdx),m_voc.row(trainIdx),residual,noArray());
        add(responseHist.row(trainIdx),residual,responseHist.row(trainIdx),noArray(),responseHist.type());
    }

    // l2-norm
    auto l2 = norm(responseHist,NORM_L2);
    responseHist /= l2;
    //normalize(responseHist,responseHist,1,0,NORM_L2);

    //Mat vec(1,m_voc.rows * f.cols,CV_32FC1,Scalar::all(0));
    vlad = responseHist.reshape(0,1); // Reshape the matrix to 1 x (k*d) vector
}

bool Vocabulary::load(const std::string &filename)
{
    FileStorage fs(filename,FileStorage::READ);

    if(!fs.isOpened()) return false;

    fs["k"] >> m_k;
    fs["voc"] >> m_voc;

    fs.release();
    return true;
}

void Vocabulary::setFeatureDetector(std::shared_ptr<RootSiftDetector> featureDetector)
{
    m_featureDetector = featureDetector;
}

bool Vocabulary::save(const std::string &filename)
{
    FileStorage fs(filename,FileStorage::WRITE);
    if(!fs.isOpened()) return false;

    fs << "k" << m_k;
    fs << "voc" << m_voc;
    
    fs.release();
    return true;
}

void  Vocabulary::getVocabulary(cv::Mat &voc) const
{
    voc = m_voc;
}

void Vocabulary::setVocabulary(cv::Mat &voc)
{
    m_voc = voc;
    m_k = m_voc.rows;
}