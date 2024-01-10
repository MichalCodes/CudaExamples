// ***********************************************************************
//
// Demo program for education in subject
// Computer Architectures and Parallel Systems.
// Petr Olivka, dep. of Computer Science, FEI, VSB-TU Ostrava, 2020/11
// email:petr.olivka@vsb.cz
//
// Example of CUDA Technology Usage.
//
// Image interface for CUDA
//
// ***********************************************************************

#pragma once

#include <opencv2/core/mat.hpp>

// Structure definition for exchanging data between Host and Device
struct CudaImg
{
    uint3 m_size;             // size of picture
    union {
        void   *m_p_void;     // data of picture
        uchar1 *m_p_uchar1;   // data of picture
        uchar3 *m_p_uchar3;   // data of picture
        uchar4 *m_p_uchar4;   // data of picture
    };

    // Function to convert 2D coordinates to 1D index
    __device__ int coord_to_index(int y, int x)
    {
        return y * m_size.x + x;
    }


    __device__ uchar1 &at1(int y, int x)
    {
        return m_p_uchar1[coord_to_index(y, x)];
    }

    __device__ uchar3 &at3(int y, int x)
    {
        return m_p_uchar3[coord_to_index(y, x)];
    }

    __device__ uchar4 &at4(int y, int x)
    {
        return m_p_uchar4[coord_to_index(y, x)];
    }
};
