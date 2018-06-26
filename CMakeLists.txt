cmake_minimum_required(VERSION 3.5)

PROJECT(image_retrieval)

set(OpenCV_DIR "/home/brook/Documents/source/opencv-3.4.1/build")

FIND_PACKAGE(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# debug
set(CMAKE_BUILD_TYPE "Debug")  
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb") 

set(CMAKE_VERBOSE_MAKEFILE on) # Output compile commandline, VERBOSE=1

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vlfeat)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/vlfeat/lib)
link_directories(/usr/lib)

aux_source_directory(. DIR_SRCS)

#add_executable(main imageRetrieval.cpp)
#add_executable(kmeans kmeans.cpp)
add_executable(bof bof_retrieval.cpp imageTrainer.cpp)

#target_link_libraries(main libvl.so ${OpenCV_LIBS}) # 
#target_link_libraries(kmeans libvl.so ${OpenCV_LIBS})
target_link_libraries(bof ${OpenCV_LIBS} libuuid.so)