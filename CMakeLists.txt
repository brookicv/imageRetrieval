cmake_minimum_required(VERSION 3.5)

PROJECT(image_retrieval)

set(OpenCV_DIR "/home/brook/Documents/source/opencv-3.4.1/build")

FIND_PACKAGE(OpenCV REQUIRED)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# debug
set(CMAKE_BUILD_TYPE "Debug")  
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb") 

set(CMAKE_VERBOSE_MAKEFILE on) # Output compile commandline, VERBOSE=1

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/vlfeat)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(/usr/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/DBow3/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/vlfeat/lib)
link_directories(/usr/lib)

ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/DBow3/src)
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/DBow3/test)
ADD_SUBDIRECTORY(${CMAKE_CURRENT_SOURCE_DIR}/src)


add_executable(main main.cpp)

target_link_libraries(main ${OpenCV_LIBS} libvl.so DBow3 retrieval)
