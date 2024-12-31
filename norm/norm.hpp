//
// Created by xiaoying on 24-11-11.
//

#ifndef CUDA_IMAGE_PROCESSING_NORM_HPP
#define CUDA_IMAGE_PROCESSING_NORM_HPP

#include <opencv2/opencv.hpp>
#include <vector>

void norm(cv::Mat& srcImg, std::vector<float>& dstImg, int height, int width, float* mean, float* std);

#endif //CUDA_IMAGE_PROCESSING_NORM_HPP
