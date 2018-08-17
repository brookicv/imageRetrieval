cmake_minimum_required(VERSION 3.5)

PROJECT(image_retrieval)

set(OpenCV_DIR "/home/brook/Documents/source/opencv-3.4.1/build")

FIND_PACKAGE(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# debug
set(CMAKE_BUILD_TYPE "Debug")  
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb") 

set(CMAKE_VERBOSE_MAKEFILE on) # Output compile commandline, VERBOSE=1

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include)

link_directories(/usr/lib)

add_executable(dbow dbow3_retrieval.cpp)
target_link_libraries(dbow ${OpenCV_LIBS} DBow3)