#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Kernel to calculate minimums of all columns in a distance matrix
__global__ void calculateColumnMinimums(
    const float* d_distances,
    int numRows,
    int numCols,
    float* d_allPba);

__global__ void calculateRowMinimums(
    const float* d_distances,
    int numRows,
    int numCols,
    float* d_allPab);

// Kernel to calculate MPDist profile from distance matrix
__global__ void calculateMPDistProfile(
    int numRows,
    int numCols,
    int k,
    float* d_allPabba,
    float* d_mpdistProfile);

__global__ void createPabba(
    const float* d_allPab,
    const float* d_allPba,
    int numRows,
    int numCols,
    float* d_allPabba);