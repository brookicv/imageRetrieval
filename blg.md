

在上一篇文章，介绍了词袋模型(Bag of Words)在图像检索中的应用。本文将简单的实现基于`BoW`的图像检索，包括以下内容：
- 基于OpenCV的实现，包括提取`sift`特征,聚类生产词典，建立kd-tree索引，检索。
- OpenCV类型`BOWTrainer`的使用
- 开源词袋模型DBow3的简单使用

### `BoW`的简单实现

`BoW`是将图像表示为`visual word`的直方图，也就是该图像包含每一个`visual word`的频数。`visual word`是图像库中具有代表性的特征向量，所以构建图像库`BoW`模型的步骤如下：
- 提取图像库中所有图像的特征，如：sifr,orb等
- 构建图像库的 `visual word`列表，也就是词典`vocabulary`。　这一步就是要提取图像库中的典型特征得到一个个`visual word`，通常是对上一步中提取到的图像特征进行聚类，得到的聚类中心就是`vocabulary`。
- 相对于`vocabulary`统计图像中特征对应于每个`visual word`的频数，

http://img.my.csdn.net/uploads/201211/29/1354200630_1518.jpg


BoW模型最初是为解决文档建模问题而提出的，因为文本本身就是由单词组成的。它通过累加单词响应到一个全局向量来给文档建立单词直方图。在图像领域，尺度不变（SIFT）特征的引入使得BoW模型变得可行。最初，SIFT由检测器和描述符组成，但现在描述符被单独提取出来使用。在这篇综述中，如果没有特别指明的话，SIFT往往是指128维的描述符（译者注：OpenCV的SIFT实现也是默认生成128维向量），这也是社区的惯例。通过一个预训练的字典（译者注：补充说明一下，在工业界的项目中，待检索的图像往往有特定的范围，使用特定范围内的有代表性的图片构建出预训练字典可以取得比较好的效果），局部特征被量化表示为视觉词汇。一张图片能够被表示成类似文档的形式，这样就可以使用经典的权重索引方案


几个术语：
- word,visual word,word表示文本处理中的一个单词，在计算机视觉中，称之为`visual word`，也就是图像的一个特征（sift，orb等）。
- vocabulary 词汇列表。

计算机视觉中的词袋模型(Bag of Words,BoW)源自于文本的处理。在文本处理中，BoW就是统计各个单词在某一篇文档出现的频数。在计算视觉中，可以将图像当作一篇文档，将从图像中提取的特征的作为一个个`Visual word`.
在使用BoW模型进行图像检索时，通常的流程如下：
- 提取图像库中所有图像的特征，如：sifr,orb等
- 构建图像库的 `visual word`列表，也就是词典`vocabulary`。 直接从图像库中提取到的特征不能直接的作为`vocabulary`

### normalize

`norm` 计算范数
`reshape` 改变矩阵的形状（包括通道个数）
 - cn	New number of channels. If the parameter is 0, the number of channels remains the same.
 - rows	New number of rows. If the parameter is 0, the number of rows remains the same. 

```
vector<double> positiveData = { 2.0, 8.0, 10.0 };
    vector<double> normalizedData_l1, normalizedData_l2, normalizedData_inf, normalizedData_minmax;
    
    // Norm to probability (total count)
    // sum(numbers) = 20.0
    // 2.0      0.1     (2.0/20.0)
    // 8.0      0.4     (8.0/20.0)
    // 10.0     0.5     (10.0/20.0)
    normalize(positiveData, normalizedData_l1, 1.0, 0.0, NORM_L1);
    
    // Norm to unit vector: ||positiveData|| = 1.0
    // 2.0      0.15
    // 8.0      0.62
    // 10.0     0.77
    normalize(positiveData, normalizedData_l2, 1.0, 0.0, NORM_L2);
    
    // Norm to max element
    // 2.0      0.2     (2.0/10.0)
    // 8.0      0.8     (8.0/10.0)
    // 10.0     1.0     (10.0/10.0)
    normalize(positiveData, normalizedData_inf, 1.0, 0.0, NORM_INF);
    
    // Norm to range [0.0;1.0]
    // 2.0      0.0     (shift to left border)
    // 8.0      0.75    (6.0/8.0)
    // 10.0     1.0     (shift to right border)
    normalipositiveData, normalizedData_minmax, 1.0, 0.0, NORM_MINMAX);
```
**注意对矩阵的归一化和对其行（列）向量归一化的区别**

- l1-norm: 
$$
    vec(i) = \frac{vec(i)}{\sum{vec(i)}}
$$

- l2-norm
$$
    vec(i) = \frac{vec(i)}{\sum{\sqrt{vec(i)^2}}
$$

```

#include "func.h"

using namespace std;
using namespace cv;

void extractFeatures(const std::vector<std::string> &imageFileList,std::vector<cv::Mat> &features)
{
    int index = 1;
    int count = 0;
    features.reserve(imageFileList.size());

    //#pragma omp parallel for
    for(const string &str : imageFileList){

        auto img = imread(str,IMREAD_GRAYSCALE);
        if(img.empty()){
            cerr << "Open image #" << str << " features failed" << endl;
            continue;
        }
        cout << "Extract feature from #" << index << "st image #" << str << endl;
        auto fdetector = xfeatures2d::SIFT::create(0,3,0.2,10);
        vector<KeyPoint> kpts;
        Mat des;
        fdetector->detectAndCompute(img,noArray(),kpts,des);
        features.emplace_back(des); 
        count += des.rows;
        index ++ ;
    }   
    cout << "Extract #" << index << "# images features done!" << "Count of features:#" << count << endl;
}

void rootSift(const cv::Mat &siftFeature,cv::Mat &rootSiftFeature)
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

void rootSift(const std::vector<cv::Mat> &siftFeatures,std::vector<cv::Mat> &rootSiftFeatures)
{   
    auto size = siftFeatures.size();
    //#pragma omp parallel for
    for(size_t i = 0; i < size; i ++){
        Mat root_sift;
        rootSift(siftFeatures[i],root_sift);
        rootSiftFeatures.push_back(root_sift);
    }
}

void Vocabulary::create(const std::vector<cv::Mat> &features,int k)
{
    Mat f;
    vconcat(features,f);
    vector<int> labes;
    kmeans(f,k,labes,TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,100,0.01),3,cv::KMEANS_PP_CENTERS,m_voc);
    m_k = k;
}

void Vocabulary::rootSift(const cv::Mat &siftFeature,cv::Mat &rootSiftFeature)
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

void Vocabulary::transform_bow(const cv::Mat &img,std::vector<int> bow)
{
    auto fdetector = xfeatures2d::SIFT::create(0,3,0.2,10);
    vector<KeyPoint> kpts;
    Mat des;
    fdetector->detectAndCompute(img,noArray(),kpts,des);

    Mat f;
    rootSift(des,f);

    // Find the nearest center
    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    vector<DMatch> matches;
    matcher->match(f,m_voc,matches);

    bow = vector<int>(m_k,0);
    // Frequency
    for( size_t i = 0; i < matches.size(); i++ ){
        int queryIdx = matches[i].queryIdx;
        int trainIdx = matches[i].trainIdx; // cluster index
        CV_Assert( queryIdx == (int)i );

        bow[trainIdx] ++; // Compute word frequency
    }

    // trainIdx => center index
    for_each(matches.begin(),matches.end(),[&bow](const DMatch &match){
        bow[match.trainIdx] ++; // Compute word frequency
    });
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

bool Vocabulary::save(const std::string &filename)
{
    FileStorage fs(filename,FileStorage::READ);
    if(!fs.isOpened()) return false;

    fs << "k" << m_k;
    fs << "voc" << m_voc;
    
    fs.release();
    return true;
}
```

https://blog.csdn.net/TTdreamloong/article/details/80991161

http://blog.sciencenet.cn/blog-713652-792911.html

https://blog.csdn.net/ttdreamloong/article/details/79216937