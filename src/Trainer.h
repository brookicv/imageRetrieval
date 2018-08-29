
#ifndef __H__TRAINER
#define __H__TRAINER

#include "Utility.h"
#include "Database.h"
#include "Vocabulary.h"
#include "RootSiftDetector.h"
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class Trainer{

public:

    Trainer();
    ~Trainer();

    Trainer(int k,int pcaDim,const std::string &imageFolder,
        const std::string &path,const std::string &identifiery,std::shared_ptr<RootSiftDetector> detector);
    
    void createVocabulary();
    void createDb();

    void save();

private:

    int m_k; // The size of vocabulary
    int m_pcaDimension; // The retrain dimensions after pca

    Vocabulary* m_voc;
    Database* m_db;

private:

    /*
        Image folder
    */
    std::string m_imageFolder;

    /*
        training result identifier,the name suffix of vocabulary and database
        voc-identifier.yml,db-identifier.yml
    */
    std::string m_identifier;

    /*
        The location of training result
    */
    std::string m_resultPath;
};

#endif