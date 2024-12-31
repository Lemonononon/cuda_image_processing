cmake_minimum_required(VERSION 3.10)

set(CMAKE_CUDA_ARCHITECTURES 86)
set(CUDA_COMPILER_PATH "/usr/local/cuda/bin/nvcc")

project(cuda_image_processing)

set(CMAKE_CXX_STANDARD 17)

# opencv
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

add_subdirectory(letterbox)
add_subdirectory(warpaffine)
add_subdirectory(resize)
add_subdirectory(norm)
add_subdirectory(cvtcolor)