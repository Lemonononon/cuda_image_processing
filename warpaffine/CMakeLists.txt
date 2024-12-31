set(CUDA_COMPILER_PATH "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_COMPILER ${CUDA_COMPILER_PATH})
enable_language(CUDA)

add_executable(warpaffine warpaffine.cpp warpaffine.cu)

target_link_libraries(warpaffine ${OpenCV_LIBS})