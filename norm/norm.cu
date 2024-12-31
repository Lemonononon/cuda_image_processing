#include "norm.hpp"
#include <cuda_runtime.h>


__global__ void norm_kernel(const uchar* src, float* dst, const int height, const int width, const float* mean, const float* std){

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = 3*(y*width+x);

    if(x < width && y < height){
        dst[idx] = ( (float)src[idx]/255.0 - mean[0]) / std[0];
        dst[idx+1] = ( (float)src[idx+1]/255.0 - mean[1]) / std[1];
        dst[idx+2] = ( (float)src[idx+2]/255.0 - mean[2]) / std[2];
    }

}



void norm(cv::Mat& srcImg, std::vector<float>& dst, int height, int width, float* mean, float* std){

    dim3 blockSize(32, 32);
    dim3 gridSize( (width+blockSize.x-1)/blockSize.x , (height+blockSize.y-1)/blockSize.y);


    uchar *d_src;
    cudaMalloc( (void**) &d_src, srcImg.total()*srcImg.elemSize()*sizeof(uchar));
    cudaMemcpy( d_src, srcImg.data, srcImg.total()*srcImg.elemSize()*sizeof(uchar), cudaMemcpyHostToDevice);

    float* d_dst;
    cudaMalloc( (void**) &d_dst, srcImg.total()*srcImg.elemSize()*sizeof(float));

    float *d_mean;
    cudaMalloc( (void**) &d_mean, 3*sizeof(float));
    cudaMemcpy( d_mean, mean, 3*sizeof(float), cudaMemcpyHostToDevice);

    float *d_std;
    cudaMalloc( (void**) &d_std, 3*sizeof(float));
    cudaMemcpy( d_std, std, 3*sizeof(float), cudaMemcpyHostToDevice);

    norm_kernel<<<gridSize, blockSize>>>(d_src, d_dst, height, width, d_mean, d_std);

    cudaMemcpy( dst.data(), d_dst, srcImg.total()*srcImg.elemSize()*sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_mean);
    cudaFree(d_std);
}