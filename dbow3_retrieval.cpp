
#include <vector>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "DBoW3.h"
#include "utils.h"

using namespace std;
using namespace cv;

void extract_features(const vector<string> &image_file_list,vector<Mat> &features){

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

void vocabulary(const vector<Mat> &features,const string &file_path,int k = 9,int l = 3){
    //Branching factor and depth levels
    const DBoW3::WeightingType weight = DBoW3::TF_IDF;
    const DBoW3::ScoringType score = DBoW3::L2_NORM;

    DBoW3::Vocabulary voc(k,l,weight,score);
    cout << "Creating a small " << k << "^" << l << " vocabulary..." << endl;
    voc.create(features);
    cout << "...done!" << endl;
    
    //cout << "Vocabulary infomation: " << endl << voc << endl << endl;

    // save the vocabulary to disk
    cout << endl << "Saving vocabulary..." << endl;
    stringstream ss;
    ss << file_path << "/small_voc.yml.gz";
    voc.save(ss.str());
    cout << "Done" << endl;
}

void database(const vector<Mat> &features,const string &file_path){
    // load the vocabulary from disk
    stringstream ss ;
    ss << file_path <<"/small_voc.yml.gz";
    DBoW3::Vocabulary voc(ss.str());

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

    // we can save the database. The created file includes the vocabulary
    // and the entries added
    cout << "Saving database..." << endl;
    db.save("small_db.yml.gz");
    cout << "... done!" << endl;
}

void save_image_file_list(const vector<string> &image_file_list,const string &data_folder){

    stringstream ss;
    ss << data_folder << "/image_list.txt";

    ofstream ofs(ss.str());
    for(const string &file : image_file_list){
        ofs << file << endl;
    }
    ofs.close();
}

void load_image_file_list(vector<string> &image_file_list,const string &data_folder){
    stringstream ss;
    ss << data_folder << "image_list.txt";

    ifstream ifs(ss.str());
    string file;
    while(ifs >>file){
        image_file_list.emplace_back(file);
    }
}

void training(const vector<string> &image_file_list,const string &data_folder,int k,int l){

    vector<Mat> features;
    //extract_features(image_file_list,features);

    //vocabulary(features,data_folder,k,l);

    //database(features,data_folder);

    save_image_file_list(image_file_list,data_folder);
}

void retrieval(const Mat &img,const DBoW3::Database &db,DBoW3::QueryResults ql,int max_resuts){
    
    auto sift = xfeatures2d::SIFT::create(0,3,0.2,10);
    vector<KeyPoint> kpts;
    Mat des;
    sift->detectAndCompute(img,noArray(),kpts,des);

    db.query(des,ql,max_resuts);
}

int main()
{
    const string image_folder = "/home/test/git/jpg";
    const string data_folder = "../data";
    const string database_name = "small_db.yml.gz";

    //stringstream ss;
    //ss << data_folder << "/" << database_name;
    //DBoW3::Database db(ss.str());

    vector<string> image_file_list;
    get_file_name_list(image_folder,image_file_list);

    int k = 9;
    int l = 3;
    training(image_file_list,data_folder,k,l);
}