#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <opencv2/flann.hpp>
#include <sys/sysinfo.h>
#include <unistd.h>

#include "DBoW3.h"
#include "utils.h"

#include <string>
extern "C" {
    #include <vl/kmeans.h>
}

#include "siftDetector.h"

using namespace std;
using namespace cv;

void cpu(int i){
    cpu_set_t mask; // cpu_set_t 代表一个cpu集合
    CPU_ZERO(&mask);
    CPU_SET(i,&mask);

    if(pthread_setaffinity_np(pthread_self(),sizeof(mask),&mask) < 0){
        cout << "Set thread affinity failed" << endl;
    }
  
    cpu_set_t get;
     
    int num = sysconf(_SC_NPROCESSORS_CONF);
    cout << "System has " << num << " processors" << endl;

    CPU_ZERO(&mask); // Clear a cpu set
    CPU_SET(i,&mask); // Put cpu_myid to mask set

    if (sched_setaffinity(0, sizeof(mask), &mask) == -1)
    {
        printf("warning: could not set CPU affinity, continuing...\n");
    }

    while (true)
    {
        CPU_ZERO(&get);
        if (sched_getaffinity(0, sizeof(get), &get) == -1)
        {
                printf("warning: cound not get cpu affinity, continuing...\n");
        }
        for (int i = 0; i < num; i++)
        {
                if (CPU_ISSET(i, &get))
                {
                        printf("this process %d is running processor : %d\n",getpid(), i);
                }
        }
    }
}

void sift_test(){
    const string file = "../1.jpg";
    Mat img = imread(file);
    SiftDetector sift_detector;

    vector<VlSiftKeypoint> kpts;
    vector<vector<float>> descriptors;
    sift_detector.detect_and_compute(img,kpts,descriptors);

    for(int i = 0; i < kpts.size(); i ++) {
        circle(img,Point(kpts[i].x,kpts[i].y),kpts[i].sigma,Scalar(0,255,0));
    }

    vector<KeyPoint> oc_kpts;
    Mat oc_des;
    Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
    sift->detectAndCompute(img,noArray(),oc_kpts,oc_des);

    for(int i = 0; i < oc_kpts.size(); i ++) {
        circle(img,oc_kpts[i].pt,oc_kpts[i].size,Scalar(255,0,0));
    }

    cout << "vl_sift:" << descriptors.size() << endl;
    cout << "opencv_sift-descriptors:" << oc_des.size() << endl;
    
    namedWindow("SIFT");
    imshow("SIFT",img);
    waitKey();
}

void extract_features(vector<string> image_file_list,vector<Mat> &features){

    int index = 1;
    int count = 0;
    for(const string &str : image_file_list){

        auto img = imread(str,IMREAD_GRAYSCALE);
        if(img.empty()){
            cerr << "Open image #" << str << " features failed" << endl;
            continue;
        }

        cout << "Extract feature from #" << index << "st image #" << str << endl;
        auto sift = xfeatures2d::SIFT::create(0,3,0.2,10);
        vector<KeyPoint> kpts;
        Mat des;
        sift->detectAndCompute(img,noArray(),kpts,des);
        features.emplace_back(des); 
        count += des.rows;
        index ++ ;
    }
    
    cout << "Extract #" << index << "# images features done!" << "Count of features:#" << count << endl;

}

void vocabulary(const vector<Mat> &features){
    //Branching factor and depth levels
    const int K = 9;
    const int L = 3;
    const DBoW3::WeightingType weight = DBoW3::TF_IDF;
    const DBoW3::ScoringType score = DBoW3::L2_NORM;

    DBoW3::Vocabulary voc(K,L,weight,score);
    cout << "Creating a small " << K << "^" << L << " vocabulary..." << endl;
    voc.create(features);
    cout << "...done!" << endl;
    
    cout << "Vocabulary infomation: " << endl << voc << endl << endl;

    /*cout << "Matching images against themselves (0 low,1 hight): " << endl;
    DBoW3::BowVector v1,v2;
    int i = 0, j = 0;
    for(const Mat &m : features) {
        voc.transform(m,v1);
        j = 0;
        for(const Mat &m1 : features){
            voc.transform(m1,v2);
            double score = voc.score(v1,v2);
            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
            j ++;
        }
        i ++;
    }*/

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    voc.save("small_voc.yml.gz");
    cout << "Done" << endl;
}

void database(const vector<Mat> &features,vector<string> &image_file_list){
    // load the vocabulary from disk
    DBoW3::Vocabulary voc("small_voc.yml.gz");

    DBoW3::Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;
    DBoW3::QueryResults ret;
    /*for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 4);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }*/
    db.query(features[0],ret,4);
    cout << "Searching for Image " << ret << endl;

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;
}


int main(int argc ,char* argv[]) 
{
    const string image_folder = "/home/test/git/imageRetrieval/images";
    vector<string> image_file_list;
    get_file_name_list(image_folder,image_file_list);
    vector<Mat> features;
    extract_features(image_file_list,features);
    vocabulary(features);
    database(features,image_file_list);
    
    return 0;
}