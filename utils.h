
#ifndef RETRIEVAL_UTILS_H
#define RETRIEVAL_UTILS_H

#include <string>
#include <vector>
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>

void get_file_name_list(const std::string &folder,std::vector<std::string> &file_list) {

    dirent *file;
    DIR* dir = opendir(folder.c_str());

    std::stringstream ss;
    while((file = readdir(dir)) != nullptr){
        if(std::string(file->d_name) == "." || std::string(file->d_name) == ".."){
            continue;
        }

        ss.str("");
        ss << folder << "/" << file->d_name;
        file_list.emplace_back(ss.str());
    }
}

#endif