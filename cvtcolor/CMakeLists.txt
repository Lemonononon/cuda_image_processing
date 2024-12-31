set(CUDA_COMPILER_PATH "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_COMPILER ${CUDA_COMPILER_PATH})
enable_language(CUDA)

add_executable(cvtcolor cvtcolor.cpp cvtcolor.cu)

target_link_libraries(cvtcolor ${OpenCV_LIBS})