// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage with unified memory.
//
// Image transformation from RGB to BW schema. 
// Image manipulation is performed by OpenCV library. 
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv {
}

// Function prototype from .cu file
void cu_flip_vertically( CudaImg input_image, CudaImg output_image );
void cu_flip_horizontally( CudaImg input_image, CudaImg output_image );
void cu_combine(CudaImg img1, CudaImg img2, CudaImg output_image, int hor_ver);
void rotate_image_90(CudaImg input_image, CudaImg output_image, int clockwise);
void cu_combine_side_by_side(CudaImg img1, CudaImg img2, CudaImg output_image);

int main( int t_numarg, char **t_arg )
{
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator( &allocator );

    if ( t_numarg < 3 )
    {
        printf( "Enter picture filename!\n" );
        return 1;
    }

    // Load image
    cv::Mat input_image = cv::imread( t_arg[ 1 ], cv::IMREAD_COLOR ); // CV_LOAD_IMAGE_COLOR );
    cv::Mat input2_image = cv::imread( t_arg[ 2 ], cv::IMREAD_COLOR ); // CV_LOAD_IMAGE_COLOR );

    cv::Mat img_alpha_canal( input_image.rows, input_image.cols + input_image.cols, CV_8UC4 ); 


    if ( !input_image.data )
    {
        printf( "Unable to read file '%s'\n", t_arg[ 1 ] );
        return 1;
    }
    // create empty image
    cv::Mat output_image( input_image.size(), CV_8UC3 );
    cv::Mat output_horizontal_image( input_image.size(), CV_8UC3 );

    cv::Mat output_combinev( input_image.size(), CV_8UC3 );
    cv::Mat output_combineh( input_image.size(), CV_8UC3 );

    cv::Mat output_rotate90cw( input_image.cols, input_image.rows, CV_8UC3 );
    cv::Mat output_rotate90ccw( input_image.cols, input_image.rows, CV_8UC3 );

    cv::Mat output_side_by_side_image( input_image.rows, input_image.cols + input2_image.cols, CV_8UC3 );

    // data for CUDA
    CudaImg input_cuda_image, output_cuda_image, output_cuda_horizontal_image;
    input_cuda_image.m_size = output_cuda_image.m_size = output_cuda_horizontal_image.m_size = make_uint3(input_image.cols, input_image.rows, input_image.channels());
    input_cuda_image.m_p_uchar3 = ( uchar3 * ) input_image.data;
    output_cuda_image.m_p_uchar3 = (uchar3 *) output_image.data;
    output_cuda_horizontal_image.m_p_uchar3 = (uchar3 *) output_horizontal_image.data;

    CudaImg output_cuda_combinev, output_cuda_combineh, input2_cuda_image;
    input2_cuda_image.m_size = output_cuda_combinev.m_size = output_cuda_combineh.m_size = make_uint3(input_image.cols, input_image.rows, input_image.channels());
    output_cuda_combinev.m_p_uchar3 = ( uchar3 * ) output_combinev.data;
    output_cuda_combineh.m_p_uchar3 = ( uchar3 * ) output_combineh.data;
    input2_cuda_image.m_p_uchar3 = ( uchar3 * ) input2_image.data;

    CudaImg output_cuda_rotate90cw, output_cuda_rotate90ccw;
    output_cuda_rotate90cw.m_size = output_cuda_rotate90ccw.m_size =  make_uint3(input_image.rows, input_image.cols, input_image.channels());
    output_cuda_rotate90cw.m_p_uchar3 = ( uchar3 * ) output_rotate90cw.data;
    output_cuda_rotate90ccw.m_p_uchar3 = ( uchar3 * ) output_rotate90ccw.data;

    // Data pro CUDA
    CudaImg output_cuda_side_by_side_image, img_cuda_alpha_canal;
    output_cuda_side_by_side_image.m_size = make_uint3(output_side_by_side_image.cols, output_side_by_side_image.rows, output_side_by_side_image.channels());
    output_cuda_side_by_side_image.m_p_uchar3 = (uchar3 *) output_side_by_side_image.data;

    // Function calling from .cu file
    cu_flip_vertically( input_cuda_image, output_cuda_image );
    cu_flip_horizontally( input_cuda_image, output_cuda_horizontal_image );
    cu_combine(input_cuda_image, input2_cuda_image, output_cuda_combineh, 1);
    cu_combine(input_cuda_image, input2_cuda_image, output_cuda_combinev, 0);
    rotate_image_90(input_cuda_image, output_cuda_rotate90cw, 0);
    rotate_image_90(input_cuda_image, output_cuda_rotate90ccw, 1);
    cu_combine_side_by_side(input_cuda_image, input2_cuda_image, output_cuda_side_by_side_image);
    // Show the input and output images
    cv::imwrite( "input.jpg", input_image );
    cv::imwrite( "output_vertical.jpg", output_image );
    cv::imwrite( "output_horizontal.jpg", output_horizontal_image );
    cv::imwrite( "output_combinev.jpg", output_combinev );
    cv::imwrite( "output_combineh.jpg", output_combineh );
    cv::imwrite( "output_rotate90cw.jpg", output_rotate90cw );
    cv::imwrite( "output_rotate90ccw.jpg", output_rotate90ccw );
    cv::imwrite( "output_side_by_side.jpg", output_side_by_side_image);
    //cv::waitKey( 0 );
}

