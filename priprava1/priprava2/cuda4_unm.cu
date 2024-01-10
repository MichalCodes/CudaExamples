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
#include <curand_kernel.h>

#include "cuda_img.h"

__global__ void kernel_alpha_blend(CudaImg img1, CudaImg img2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= img1.m_size.x || y >= img1.m_size.y)
        return;

    float a = img2.at4(x, y).w / 255.0f; // Use alpha channel of img2
    float b = 1.0f - a;

    img1.at3(x, y) = make_uchar3(static_cast<unsigned char>(a * img2.at4(x, y).x + b * img1.at3(x, y).x),
                                 static_cast<unsigned char>(a * img2.at4(x, y).y + b * img1.at3(x, y).y),
                                 static_cast<unsigned char>(a * img2.at4(x, y).z + b * img1.at3(x, y).z));
}
__global__ void kernel_reduce_rgb(CudaImg img)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= img.m_size.x || y >= img.m_size.y)
        return;

    uchar3 pixel = img.at3(y, x);

    // reduce all 3 color channels by half
    pixel.x /= 2;
    pixel.y /= 2;
    pixel.z /= 2;

    img.at3(y, x) = pixel;
}
__global__ void kernel_double_image(CudaImg old_img, CudaImg new_img, int option)
{
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int x = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= old_img.m_size.y || y >= old_img.m_size.x)
        return;

    int newY = 2 * y;
    int newX = 2 * x;
    if (option == 1)
    {
        new_img.at3(y, newX) = old_img.at3(y, x);
        new_img.at3(y, newX + 1) = old_img.at3(y, x);
    }
    else if (option == 2)
    {
        new_img.at3(newY, x) = old_img.at3(y, x);
        new_img.at3(newY + 1, x) = old_img.at3(y, x);
    }
    else
    {
        new_img.at3(newY, newX) = old_img.at3(y, x);
        new_img.at3(newY + 1, newX) = old_img.at3(y, x);
        new_img.at3(newY, newX + 1) = old_img.at3(y, x);
        new_img.at3(newY + 1, newX + 1) = old_img.at3(y, x);
    }
}

__global__ void kernel_half_image(CudaImg old_img, CudaImg new_img, int option)
{
    int y = blockDim.x * blockIdx.x + threadIdx.x;
    int x = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= old_img.m_size.y || y >= old_img.m_size.x)
        return;

    if (option == 1 && x % 2 == 0)
        new_img.at3(y, x / 2) = old_img.at3(y, x);
    else if (option == 2 && y % 2 == 0)
        new_img.at3(y / 2, x) = old_img.at3(y, x);
    else if (option == 3 && x % 2 == 0 && y % 2 == 0)
        new_img.at3(y / 2, x / 2) = old_img.at3(y, x);
}


__global__ void kernel_chessboard(CudaImg img)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // If pixel is outside of the image boundaries, return
    if (x >= img.m_size.x || y >= img.m_size.y)
        return;

    // Compute block coordinates
    int block_x = x / blockDim.x;
    int block_y = y / blockDim.y;

    // Compute color based on block coordinates
    uchar3 color;
    if ((block_x + block_y) % 3 == 0)
        color = make_uchar3(255, 0, 0); // red
    else if ((block_x + block_y) % 3 == 1)
        color = make_uchar3(0, 255, 0); // green
    else
        color = make_uchar3(0, 0, 255); // blue

    img.at3(x, y) = color;
}
void cu_alpha_blend(CudaImg img1, CudaImg img2)
{
    cudaError_t l_cerr;
    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 32;
    dim3 l_blocks((img1.m_size.x + l_block_size - 1) / l_block_size,
                  (img1.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_alpha_blend<<<l_blocks, l_threads>>>(img1, img2);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

void cu_chessboard(CudaImg cudaimg){
       cudaError_t l_cerr;
    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 32;
    dim3 l_blocks((cudaimg.m_size.x + l_block_size - 1) / l_block_size,
                  (cudaimg.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_chessboard<<<l_blocks, l_threads>>>(cudaimg);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}


void cu_double_image(CudaImg old_pic, CudaImg new_pic, int option)
{
    cudaError_t l_cerr;
    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks((old_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (old_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_double_image<<<l_blocks, l_threads>>>(old_pic, new_pic, option);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

void cu_half_image(CudaImg old_pic, CudaImg new_pic, int option)
{
    cudaError_t l_cerr;
    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks((old_pic.m_size.x + l_block_size - 1) / l_block_size,
                  (old_pic.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_half_image<<<l_blocks, l_threads>>>(old_pic, new_pic, option);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}

void cu_reduce_rgb(CudaImg img)
{
    cudaError_t l_cerr;
    // Grid creation, size of grid must be equal or greater than images
    int l_block_size = 16;
    dim3 l_blocks((img.m_size.x + l_block_size - 1) / l_block_size,
                  (img.m_size.y + l_block_size - 1) / l_block_size);
    dim3 l_threads(l_block_size, l_block_size);

    kernel_reduce_rgb<<<l_blocks, l_threads>>>(img);

    if ((l_cerr = cudaGetLastError()) != cudaSuccess)
        printf("CUDA Error [%d] - '%s'\n", __LINE__, cudaGetErrorString(l_cerr));

    cudaDeviceSynchronize();
}
