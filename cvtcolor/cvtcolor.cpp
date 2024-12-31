//
// Created by xiaoying on 24-11-11.
//

#include "cvtcolor.hpp"

int main( int argc, char** argv ){

    cv::Mat srcImg = cv::imread(argv[1]);

    // std::cout << srcImg.total() << std::endl;
    // std::cout << srcImg.elemSize() << std::endl;

    cv::Mat cvtColorImg(srcImg.rows, srcImg.cols, CV_8UC3);

    cvtColor(srcImg, cvtColorImg);

    cv::imwrite("cvtColorImg.jpg", cvtColorImg);


}