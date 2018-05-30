cmake_minimum_required(VERSION 3.10)

PROJECT(image_retrieval)

set(OpenCV_DIR "/home/brook/Documents/source/opencv-3.4.1/build")

FIND_PACKAGE(OpenCV REQUIRED)

# debug
set(CMAKE_BUILD_TYPE "Debug")  
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb") 

set(CMAKE_VERBOSE_MAKEFILE on) # Output compile commandline, VERBOSE=1

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vlfeat)
include_directories(${OpenCV_INCLUDE_DIRS})

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/vlfeat/lib)

aux_source_directory(. DIR_SRCS)

add_executable(main ${DIR_SRCS})
target_link_libraries(main libvl.so ${OpenCV_LIBS}) # 