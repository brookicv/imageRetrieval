cmake_minimum_required(VERSION 3.5)

PROJECT(image_retrieval)

set(OpenCV_DIR "/home/brook/Documents/source/opencv-3.4.1/build")

FIND_PACKAGE(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_INSTALL_PREFIX "{CMAKE_CURRENT_SOURCE_DIR}")

# debug
set(CMAKE_BUILD_TYPE "Debug")  
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb") 

set(CMAKE_VERBOSE_MAKEFILE on) # Output compile commandline, VERBOSE=1


aux_source_directory(. DIR_SRCS)

ADD_LIBRARY (DBow3 SHARED ${DIR_SRCS})
target_link_libraries(DBow3 ${OpenCV_LIBS})


