//
// Created by xiaoying on 24-11-11.
//

#ifndef CUDA_IMAGE_PROCESSING_RESIZE_HPP
#define CUDA_IMAGE_PROCESSING_RESIZE_HPP

#include <opencv2/opencv.hpp>
void resize(const cv::Mat& srcImg, cv::Mat& dstImg, const int dstHeight, const int dstWidth);

#endif //CUDA_IMAGE_PROCESSING_RESIZE_HPP
