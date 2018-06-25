
#include "utils.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "imageTrainer.h"


using namespace std;
using namespace cv;

int main()
{
    const string image_folder = "/home/brook/git_folder/image_retrieval/images";
    vector<string> image_file_list;
    get_file_name_list(image_folder,image_file_list);

    ImageTrainer trainer(200,image_file_list);

    trainer.extract_sift();
    trainer.vocabulary_kmeans();

    
    return 0;
}