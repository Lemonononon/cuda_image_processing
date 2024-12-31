//
// Created by xiaoying on 24-11-11.
//

#include "norm.hpp"

int main( int argc, char** argv ){

    if (argc != 2){
        std::cout << "Usage: " << argv[0] << " <image path>" << std::endl;
        return -1;
    }

    cv::Mat srcImg = cv::imread(argv[1]);
    std::vector<float> dst(srcImg.elemSize()*srcImg.total());
    std::vector<float> mean = {0, 0, 0};
    std::vector<float> std = {1, 1, 1};

    norm( srcImg, dst, srcImg.rows, srcImg.cols, mean.data(), std.data() );

    for (int i = 0; i < 20; ++i) {
        std::cout << dst[i] << " ";
    }

    std::cout << std::endl;

}