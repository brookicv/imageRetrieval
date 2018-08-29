
#include "Trainer.h"

using namespace std;
using namespace cv;

Trainer::Trainer(){}
Trainer::~Trainer()
{
    if(m_voc) delete m_voc;
    if(m_db)  delete m_db;
}

Trainer::Trainer(int k,int pcaDim,const std::string &imageFolder,
        const std::string &path,const std::string &identifiery,std::shared_ptr<RootSiftDetector> detector)
{
   m_k              = k;
   m_pcaDimension   = pcaDim;
   m_imageFolder    = imageFolder;
   m_resultPath     = path;
   m_identifier     = identifiery;

    m_voc           = new Vocabulary(detector,m_k);
    m_db            = new Database(detector);
}

void Trainer::createVocabulary()
{   
    vector<string> fileList;
    PathManager::get_path_list(m_imageFolder,fileList);

    m_voc->create(fileList);
}

void Trainer::createDb()
{
    vector<string> fileList;
    PathManager::get_path_list(m_imageFolder,fileList);
    m_db->setVocabulary(*m_voc);

    m_db->buildDatabase(fileList,m_pcaDimension);
}

void Trainer::save()
{
    stringstream ss;
    ss << m_resultPath << "/voc-" << m_identifier << ".yml";
    m_voc->save(ss.str());

    ss.str("");
    m_db->save1(m_resultPath,m_identifier);
}