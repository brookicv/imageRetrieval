
#include "utils.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "imageTrainer.h"

#include <opencv2/flann/kdtree_index.h>


using namespace std;
using namespace cv;

void kd_test(){

    // Build kd tree
    Point2f p1(2,3),   p2(5,4),p3(9, 6),  p4(4,7),  p5(8,1),p6(7, 2);
    vector<cv::Point2f> source;
    source.push_back(p1);
    source.push_back(p2);
    source.push_back(p3);
    source.push_back(p4);
    source.push_back(p5);
    source.push_back(p6);
    source.push_back(Point2f(1,1.5));
    flann::KDTreeIndexParams indexParams(2);
    flann::Index kdtree(cv::Mat(source).reshape(1), indexParams);

    // Query
    vector<float> query;
    Point2f pt(0.1,1);
    query.push_back(pt.x);
    query.push_back(pt.y);
    int k = 4; //number of nearest neighbors
    vector<int> indices(k);//找到点的索引
    vector<float> dists(k);

    flann::SearchParams params(128);
    kdtree.knnSearch(query, indices, dists, k,params);
    cout<<indices[0]<< " : dist->" << dists[0] << " point:" << source[indices[0]] << endl;
    cout<<indices[1]<<" : dist->" << dists[1] << " point:" << source[indices[1]] << endl;
    cout<<indices[2]<<" : dist->" << dists[2] << " point:" << source[indices[2]] << endl;
    cout<<indices[3]<<" : dist->" << dists[3] << " point:" << source[indices[3]] << endl;
}

int main()
{
    /*const string image_folder = "/home/brook/git_folder/image_retrieval/images";
    vector<string> image_file_list;
    get_file_name_list(image_folder,image_file_list);

    ImageTrainer trainer(200,image_file_list);

    trainer.extract_sift();
    trainer.vocabulary_kmeans();*/

    kd_test();
    
    
    return 0;
}