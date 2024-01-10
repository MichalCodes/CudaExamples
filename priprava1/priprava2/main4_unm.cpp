#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include "uni_mem_allocator.h"
#include "cuda_img.h"

namespace cv {
}

// Function prototype from .cu file
void cu_double_image(CudaImg old_pic, CudaImg new_pic, int option); //roztazeni obrazku po osach x nebo y nebo x a y.
void cu_reduce_rgb(CudaImg img); //zmensi vsechny rgb o polovinu.
void cu_chessboard(CudaImg cudaimg);
void cu_half_image(CudaImg old_pic, CudaImg new_pic, int option);
void cu_alpha_blend(CudaImg img1, CudaImg img2);

int main(int t_numarg, char **t_arg) {
    // Uniform Memory allocator for Mat
    UniformAllocator allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    if (t_numarg < 2) {
        printf("Enter picture filename!\n");
        return 1;
    }

    // Load image
    cv::Mat input_image = cv::imread(t_arg[1], cv::IMREAD_COLOR);

    if (!input_image.data) {
        printf("Unable to read file '%s'\n", t_arg[1]);
        return 1;
    }

    // create empty image with the same number of channels as the input image
    cv::Mat output_double_image(2 * input_image.rows, 2 * input_image.cols, CV_8UC3);
    cv::Mat output_doubleX_image(input_image.rows, 2 * input_image.cols, CV_8UC3);
    cv::Mat output_doubleY_image(2 * input_image.rows,input_image.cols, CV_8UC3);
    cv::Mat output_chessboard(input_image.rows,input_image.cols, CV_8UC3);
    cv::Mat half_output(input_image.rows,input_image.cols, CV_8UC3);
    cv::Mat half_outputX(input_image.rows,input_image.cols, CV_8UC3);
    // create empty image with half the size of the input image in the Y direction
    cv::Mat output_halfY_image(input_image.rows / 2, input_image.cols, CV_8UC3);
    cv::Mat output_halfX_image(input_image.rows, input_image.cols / 2, CV_8UC3);
    cv::Mat output_half_image(input_image.rows / 2, input_image.cols / 2, CV_8UC3);

    // Vytvoření prázdného obrázku s 4 kanály (BGR + alfa)
    cv::Mat red_image(input_image.rows, input_image.cols, CV_8UC4);

    // Nastavení červené barvy a alfa kanálu
    cv::Vec4b red_color(0, 0, 255, 50); // BGR + alfa

    // Vyplnění obrázku červenou barvou
    for (int y = 0; y < input_image.rows; y++)
    {
        for (int x = 0; x < input_image.cols; x++)
        {
            red_image.at<cv::Vec4b>(y, x) = red_color;
        }
    }


    // data for CUDA
    CudaImg output_cuda_halfY_image;
    output_cuda_halfY_image.m_size.x = output_halfY_image.size().width;
    output_cuda_halfY_image.m_size.y = output_halfY_image.size().height;
    output_cuda_halfY_image.m_p_uchar3 = (uchar3 *)output_halfY_image.data;

    // data for CUDA
    CudaImg output_cuda_halfX_image;
    output_cuda_halfX_image.m_size.x = output_halfX_image.size().width;
    output_cuda_halfX_image.m_size.y = output_halfX_image.size().height;
    output_cuda_halfX_image.m_p_uchar3 = (uchar3 *)output_halfX_image.data;

    // data for CUDA
    CudaImg output_cuda_half_image;
    output_cuda_half_image.m_size.x = output_half_image.size().width;
    output_cuda_half_image.m_size.y = output_half_image.size().height;
    output_cuda_half_image.m_p_uchar3 = (uchar3 *)output_half_image.data;




    // data for CUDA
    CudaImg input_cuda_image;
    input_cuda_image.m_size.x = input_image.size().width;
    input_cuda_image.m_size.y = input_image.size().height;
    input_cuda_image.m_p_uchar3 = (uchar3 *)input_image.data;

    CudaImg output_cuda_double_image;
    output_cuda_double_image.m_size.x = output_double_image.size().width;
    output_cuda_double_image.m_size.y = output_double_image.size().height;
    output_cuda_double_image.m_p_uchar3 = (uchar3 *)output_double_image.data;

    
    CudaImg output_cuda_doubleX_image;
    output_cuda_doubleX_image.m_size.x = output_doubleX_image.size().width;
    output_cuda_doubleX_image.m_size.y = output_doubleX_image.size().height;
    output_cuda_doubleX_image.m_p_uchar3 = (uchar3 *)output_doubleX_image.data;

    CudaImg output_cuda_chessboard;
    output_cuda_chessboard.m_size.x = output_chessboard.size().width;
    output_cuda_chessboard.m_size.y = output_chessboard.size().height;
    output_cuda_chessboard.m_p_uchar3 = (uchar3 *)output_chessboard.data;

    
    CudaImg output_cuda_doubleY_image;
    output_cuda_doubleY_image.m_size.x = output_doubleY_image.size().width;
    output_cuda_doubleY_image.m_size.y = output_doubleY_image.size().height;
    output_cuda_doubleY_image.m_p_uchar3 = (uchar3 *)output_doubleY_image.data;

    CudaImg red_cuda_image;
    red_cuda_image.m_size.x = red_image.size().width;
    red_cuda_image.m_size.y = red_image.size().height;
    red_cuda_image.m_p_uchar4 = (uchar4 *)red_image.data;

    // Function calling from .cu file
    cu_double_image(input_cuda_image, output_cuda_doubleX_image, 1);
    cu_double_image(input_cuda_image, output_cuda_doubleY_image, 2);
    cu_double_image(input_cuda_image, output_cuda_double_image, 3);
    cu_half_image(input_cuda_image, output_cuda_halfX_image, 1);
    cu_half_image(input_cuda_image, output_cuda_halfY_image, 2);
    cu_half_image(input_cuda_image, output_cuda_half_image, 3);
    cu_alpha_blend(input_cuda_image, red_cuda_image);

    // Show the input and output images
    cv::imwrite("input.jpg", input_image);
    cv::imwrite("output_double_image.png", output_double_image);
    cv::imwrite("output_doubleX_image.png", output_doubleX_image);
    cv::imwrite("output_doubleY_image.png", output_doubleY_image);
    cu_reduce_rgb(input_cuda_image);
    cv::imwrite("reduced_RGB_image.png", input_image);
    cu_chessboard( output_cuda_chessboard);
    cv::imwrite("chess_board.jpg",  output_chessboard);
    cv::imwrite("output_halfY_image.png", output_halfY_image);
    cv::imwrite("output_half_image.png", output_half_image);
    cv::imwrite("output_halfX_image.png", output_halfX_image);
    





}
