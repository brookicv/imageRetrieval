#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <iostream>
#include <omp.h>
#include <sys/stat.h>
#include <dirent.h>

#include "Vocabulary.h"
#include "Utility.h"
#include "Database.h"

#include "Trainer.h"
#include "RootSiftDetector.h"
#include <memory>

using namespace std;
using namespace cv;


int main(int argc, char *argv[])
{
    const string image_200 = "/home/test/images-1";
    const string image_6k = "/home/test/images/sync_down_1";
    
    auto detector = make_shared<RootSiftDetector>(5,5,10);
    Trainer trainer(64,0,image_200,"/home/test/projects/imageRetrievalService/build","test-200-vl-64",detector);

    trainer.createVocabulary();
    trainer.createDb();
    
    trainer.save();

    return 0;
}
