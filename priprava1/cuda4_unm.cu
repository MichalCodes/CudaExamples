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
//
// ***********************************************************************

#include <stdio.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include "cuda_img.h"
__global__ void kernel_rotate_image_90(CudaImg input_image, CudaImg output_image, int clockwise)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < input_image.m_size.x && y < input_image.m_size.y)
    {
        uchar3 input_pixel = input_image.at3(y, x);

        int new_x, new_y;
        if (clockwise)
        {
            new_x = y;
            new_y = input_image.m_size.x - x - 1;
        }
        else
        {
            new_x = input_image.m_size.y - y - 1;
            new_y = x;
        }

        output_image.at3(new_y, new_x) = input_pixel;
    }
}
// Demo kernel to transform RGB color schema to BW schema
__global__ void kernel_vertical_flip(CudaImg input_image, CudaImg output_image)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < input_image.m_size.x && y < input_image.m_size.y)
    {
        uchar3 input_pixel = input_image.at3(y, x);
        int output_index = (input_image.m_size.y - y - 1) * input_image.m_size.x + x;
        output_image.m_p_uchar3[output_index] = input_pixel;
    }
}
__global__ void kernel_combine_images_side_by_side(CudaImg img1, CudaImg img2, CudaImg output_image)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y < img1.m_size.y && x < (img1.m_size.x + img2.m_size.x))
    {
        uchar3 input_pixel;

        if (x < img1.m_size.x)
            input_pixel = img1.at3(y, x);
        else
            input_pixel = img2.at3(y, x - img1.m_size.x);

        output_image.at3(y, x) = input_pixel;
    }
}


__global__ void kernel_join_images_side_by_side(CudaImg img1, CudaImg img2, CudaImg output_image)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < img1.m_size.x * 2 && y < img1.m_size.y)
    {
        uchar3 input_pixel;

        if (x < img1.m_size.x)
            input_pixel = img1.at3(y, x);
        else
            input_pixel = img2.at3(y, x - img1.m_size.x);

        output_image.at3(y, x) = input_pixel;
    }
}

__global__ void kernel_overlay_image_with_transparency(CudaImg base_image, CudaImg overlay_image, CudaImg output_image, float alpha)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < base_image.m_size.x && y < base_image.m_size.y)
    {
        uchar3 base_pixel = base_image.at3(y, x);
        uchar3 overlay_pixel = overlay_image.at3(y, x);

        uchar3 output_pixel;
        output_pixel.x = base_pixel.x * (1 - alpha) + overlay_pixel.x * alpha;
        output_pixel.y = base_pixel.y * (1 - alpha) + overlay_pixel.y * alpha;
        output_pixel.z = base_pixel.z * (1 - alpha) + overlay_pixel.z * alpha;

        output_image.at3(y, x) = output_pixel;
    }
}


__global__ void kernel_combine_images(CudaImg img1, CudaImg img2, CudaImg output_image, int hor_ver)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < img1.m_size.x && y < img1.m_size.y)
    {
        uchar3 input_pixel;

        if (hor_ver == 0) // Horizontal
        {
            if (x < img1.m_size.x / 2)
                input_pixel = img1.at3(y, x);
            else
                input_pixel = img2.at3(y, x);
        }
        else // Vertical
        {
            if (y < img1.m_size.y / 2)
                input_pixel = img1.at3(y, x);
            else
                input_pixel = img2.at3(y, x);
        }

        output_image.at3(y, x) = input_pixel;
    }
}

// Demo kernel to perform a horizontal flip
__global__ void kernel_horizontal_flip(CudaImg input_image, CudaImg output_image)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < input_image.m_size.x && y < input_image.m_size.y)
    {
        uchar3 input_pixel = input_image.at3(y, x);
        int output_index = y * input_image.m_size.x + (input_image.m_size.x - x - 1);
        output_image.m_p_uchar3[output_index] = input_pixel;
    }
}
__global__ void kernel_add_image_with_alpha(CudaImg base_image, CudaImg alpha_image, CudaImg output_image)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (y < base_image.m_size.y && x < base_image.m_size.x)
    {
        uchar3 base_pixel = base_image.at3(y, x);
        uchar4 alpha_pixel = alpha_image.at4(y, x);

        uchar3 output_pixel;
        float alpha = alpha_pixel.w / 255.0f;

        output_pixel.x = base_pixel.x * (1.0f - alpha) + alpha_pixel.x * alpha;
        output_pixel.y = base_pixel.y * (1.0f - alpha) + alpha_pixel.y * alpha;
        output_pixel.z = base_pixel.z * (1.0f - alpha) + alpha_pixel.z * alpha;

        output_image.at3(y, x) = output_pixel;
    }
}


void cu_flip_vertically(CudaImg input_image, CudaImg output_image)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((input_image.m_size.x + l_block_size - 1) / l_block_size, (input_image.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_vertical_flip<<<l_blocks, l_threads>>>(input_image, output_image); // <-- Update the function name here

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}
void cu_flip_horizontally(CudaImg input_image, CudaImg output_image)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((input_image.m_size.x + l_block_size - 1) / l_block_size, (input_image.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_horizontal_flip<<<l_blocks, l_threads>>>(input_image, output_image);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}
void cu_combine(CudaImg img1, CudaImg img2, CudaImg output_image, int hor_ver)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((img1.m_size.x + l_block_size - 1) / l_block_size, (img1.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_combine_images<<<l_blocks, l_threads>>>(img1, img2, output_image, hor_ver);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}
void rotate_image_90(CudaImg input_image, CudaImg output_image, int clockwise)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((input_image.m_size.x + l_block_size - 1) / l_block_size, (input_image.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_rotate_image_90<<<l_blocks, l_threads>>>(input_image, output_image, clockwise);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}
void cu_combine_side_by_side(CudaImg img1, CudaImg img2, CudaImg output_image)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((img1.m_size.x + img2.m_size.x + l_block_size - 1) / l_block_size, (img1.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_combine_images_side_by_side<<<l_blocks, l_threads>>>(img1, img2, output_image);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}
void cu_add_image_with_alpha(CudaImg base_image, CudaImg alpha_image, CudaImg output_image)
{
    cudaError_t l_cerr;

    int l_block_size = 16;
    dim3 l_blocks((base_image.m_size.x + l_block_size - 1) / l_block_size, (base_image.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);
    kernel_add_image_with_alpha<<<l_blocks, l_threads>>>(base_image, alpha_image, output_image);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}


