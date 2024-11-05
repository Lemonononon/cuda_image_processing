#include <opencv2/opencv.hpp>
#include "letterbox.hpp"

int main(){

    cv::Mat img = cv::imread("../test.jpg");


    cv::Mat letterbox_cuda_img(640, 640, CV_8UC3);

    // letterbox_cuda(img, letterbox_cuda_img, target_size, color);

    letterbox(img, letterbox_cuda_img, 640, 640);

    cv::imwrite("letterbox.jpg", letterbox_cuda_img);

}