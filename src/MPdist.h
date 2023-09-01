/*+++++++++++++++++++++++++++++++++
Project: PSF (Parallel Snippet-Finder)
Source file: MPdist.h
Purpose: Parallel implementation of the MPdist measure in CUDA
Author(s): Andrey Goglachev (goglachevai@susu.ru)
+++++++++++++++++++++++++++++++++*/

#pragma once

void MPdist_(float *d_distance_matrix, float *d_profile, int n, int m, int l, int idx);

__global__ void computeMPdist(float *d_Pab, float *d_Pba, float *d_MPdist, int n, int l, int idx);

__global__ void precompute_min_Pab(float *d_distance_matrix, float *d_Pab, int n, int l);

__global__ void computePab_(float *d_distance_matrix, float *d_Pab, int n, int l, float *L_matrix, float *R_matrix);

__global__ void computePba_(float *d_distance_matrix, float *d_Pba, int n, int l);

__device__ void warpReduce(volatile float *sdata, unsigned int tid, int l);
