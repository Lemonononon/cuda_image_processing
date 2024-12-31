//
// Created by xiaoying on 24-11-11.
//
#include "resize.hpp"

int main( int argc, char** argv ){

    if (argc != 2){
        std::cout << "Usage: " << argv[0] << " <image path>" << std::endl;
        return -1;
    }

    cv::Mat srcImg = cv::imread(argv[1]);

    cv::Mat dstImg;

    // enum InterpolationFlags{
    //     /** nearest neighbor interpolation */
    //     INTER_NEAREST        = 0,
    //     /** bilinear interpolation */
    //     INTER_LINEAR         = 1,
    //     /** bicubic interpolation */
    //     INTER_CUBIC          = 2,
    //     /** resampling using pixel area relation. It may be a preferred method for image decimation, as
    //     it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST
    //     method. */
    //     INTER_AREA           = 3,
    //     /** Lanczos interpolation over 8x8 neighborhood */
    //     INTER_LANCZOS4       = 4,
    //     /** Bit exact bilinear interpolation */
    //     INTER_LINEAR_EXACT = 5,
    //     /** mask for interpolation codes */
    //     INTER_MAX            = 7,
    //     /** flag, fills all of the destination image pixels. If some of them correspond to outliers in the
    //     source image, they are set to zero */
    //     WARP_FILL_OUTLIERS   = 8,
    //     /** flag, inverse transformation
    //
    //     For example, #linearPolar or #logPolar transforms:
    //     - flag is __not__ set: \f$dst( \rho , \phi ) = src(x,y)\f$
    //     - flag is set: \f$dst(x,y) = src( \rho , \phi )\f$
    //     */
    //     WARP_INVERSE_MAP     = 16
    // };

    cv::resize(srcImg, dstImg, cv::Size(300, 300), 0, 0, cv::INTER_LINEAR);


    cv::Mat dstImgCuda;
    resize(srcImg, dstImgCuda, 300, 300);



}