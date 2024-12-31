set(CUDA_COMPILER_PATH "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_COMPILER ${CUDA_COMPILER_PATH})
enable_language(CUDA)

add_executable(resize resize.cpp resize.cu)

target_link_libraries(resize ${OpenCV_LIBS})