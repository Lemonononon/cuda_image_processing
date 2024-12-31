#include "cvtcolor.hpp"
#include <cuda_runtime.h>

// BGR2RGB
__global__ void cvtColorKernel(const uchar* src, uchar* dst, const int height, const int width){

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    auto idx = 3*(y*width+x);

    if(x < width && y < height){
        dst[idx] = src[idx+2];
        dst[idx+1] = src[idx+1];
        dst[idx+2] = src[idx];
    }
}

void cvtColor( const cv::Mat& srcImg, cv::Mat& dstImg ){
    dim3 blockSize(32, 32);
    dim3 gridSize( (srcImg.cols-1)/blockSize.x+1, (srcImg.rows-1)/blockSize.y+1 );

    uchar *src_d, *dst_d;

    cudaMalloc((void **) &src_d, srcImg.total() * srcImg.elemSize() * sizeof(uchar) );
    cudaMemcpy( src_d, srcImg.data, srcImg.total()*srcImg.elemSize()* sizeof(uchar), cudaMemcpyHostToDevice );


    cudaMalloc( (void**) &dst_d, srcImg.total()*srcImg.elemSize()* sizeof(uchar));

    cvtColorKernel<<<gridSize, blockSize>>>(src_d, dst_d, srcImg.rows, srcImg.cols);

    cudaMemcpy( dstImg.data, dst_d, srcImg.total() * srcImg.elemSize() * sizeof(uchar), cudaMemcpyDeviceToHost );


    cudaFree(dst_d);
    cudaFree(src_d);

}