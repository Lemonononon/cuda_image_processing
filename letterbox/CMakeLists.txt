
set(CUDA_COMPILER_PATH "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_COMPILER ${CUDA_COMPILER_PATH})
enable_language(CUDA)

add_executable(letterbox letterbox.cpp letterbox.cu)

target_link_libraries(letterbox ${OpenCV_LIBS})

