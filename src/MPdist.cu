/*+++++++++++++++++++++++++++++++++
Project: PSF (Parallel Snippet-Finder)
Source file: MPdist.cu
Purpose: Parallel implementation of the MPdist measure in CUDA
Author(s): Andrey Goglachev (goglachevai@susu.ru)
+++++++++++++++++++++++++++++++++*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "MPdist.h"
// #include "common.h"
#define IDX2F(i, j, n) (i * n + j)
#include <limits>
#include <cmath>

void MPdist_(float *d_distance_matrix, float *d_profile, int n, int m, int l, int idx)
{
	float *d_Pab;
	cudaMalloc(&d_Pab, l * (n - l) * sizeof(float));
	float *d_Pba;
	cudaMalloc(&d_Pba, (n - l) * sizeof(float));

	float *L_matrix;
	cudaMalloc(&L_matrix, l * l * sizeof(float));
	float *R_matrix;
	cudaMalloc(&R_matrix, (l + 1) * l * sizeof(float));

	// computePab_<<<dim3(n - l, l), 256, l * sizeof(float) / 2>>>(d_distance_matrix, d_Pab, n - l, l);
	precompute_min_Pab<<<l, 256, l * sizeof(float) / 2>>>(d_distance_matrix, d_Pab, n - l, l);
	cudaDeviceSynchronize();
	computePab_<<<1, 256>>>(d_distance_matrix, d_Pab, n - l, l, L_matrix, R_matrix);
	cudaDeviceSynchronize();
	computePba_<<<n - l, 256, l * sizeof(float) / 2>>>(d_distance_matrix, d_Pba, n - l, l);
	cudaDeviceSynchronize();
	computeMPdist<<<n - l, 256, 2 * l * sizeof(float)>>>(d_Pab, d_Pba, d_profile, n - l, l, idx);
	cudaDeviceSynchronize();

	cudaFree(d_Pab);
	cudaFree(d_Pba);
	cudaFree(L_matrix);
	cudaFree(R_matrix);
}

__global__ void computeMPdist(float *d_Pab, float *d_Pba, float *d_MPdist, int n, int l, int idx)
{
	unsigned int tid = threadIdx.x;
	extern __shared__ float sdata[];
	if (tid < l / 2)
	{
		sdata[tid] = d_Pab[tid + l * blockIdx.x];
		sdata[tid + l] = d_Pba[tid + blockIdx.x];
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2)
	{
		if (tid < stride && tid + stride < l)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + stride]);
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		warpReduce(sdata, tid, l);
	}

	if (tid == 0)
	{
		// printf("%d: %f\n", blockIdx.x, sdata[0]);
		d_MPdist[blockIdx.x + idx * (n - 2 * l)] = sdata[0];
	}
}

__global__ void precompute_min_Pab(float *d_distance_matrix, float *d_Pab, int n, int l)
{
	unsigned int tid = threadIdx.x;
	extern __shared__ float sdata[];

	if (tid < l / 2)
	{
		sdata[tid] = min(d_distance_matrix[blockIdx.x * n + tid], d_distance_matrix[blockIdx.x * n + tid + l / 2]);
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2)
	{
		if (tid < stride && tid + stride < l / 2)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + stride]);
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		warpReduce(sdata, tid, l);
	}

	if (tid == 0)
	{
		d_Pab[blockIdx.x] = sdata[0];
	}
}

__global__ void computePab_(float *d_distance_matrix, float *d_Pab, int n, int l, float *L_matrix, float *R_matrix)
{
	// sdata[tid] = min(d_distance_matrix[blockIdx.y * n + tid + blockIdx.x], d_distance_matrix[blockIdx.y * n + tid + l / 2 + blockIdx.x]);
	unsigned int tid = threadIdx.x;

	float m = d_Pab[tid];
	if (tid < l)
	{
		float *L = &L_matrix[tid * l];
		float *R = &R_matrix[tid * (l + 1)];
		for (int i = 0; i < l; i++)
			R[i] = d_distance_matrix[n * tid + i];
		int len_L = 0;
		int len_R = l;
		float R_min = d_Pab[tid];
		float mn = INFINITY;

		for (int i = l; i < n; i++)
		{
			R[len_R] = d_distance_matrix[n * tid + i];
			R_min = min(R_min, R[len_R]);
			len_R++;

			if (len_L == 0)
			{
				mn = INFINITY;
				for (int j = len_R - 1; j >= 0; j--)
				{
					mn = min(mn, R[j]);
					L[len_L] = mn;
					len_L++;
				}
				len_R = 0;
				R_min = INFINITY;
			}
			len_L--;

			// if (tid == 0) printf("%d %d: %f\n", i, tid, min(R_min, L[len_L]));
			d_Pab[l * i + tid] = min(R_min, L[len_L]);
		}
	}
}

__global__ void computePba_(float *d_distance_matrix, float *d_Pba, int n, int l)
{
	unsigned int tid = threadIdx.x;
	extern __shared__ float sdata[];

	if (tid < l / 2)
	{
		sdata[tid] = min(d_distance_matrix[tid * n + blockIdx.x], d_distance_matrix[(tid + l / 2) * n + blockIdx.x]);
	}
	__syncthreads();

	for (unsigned int stride = blockDim.x / 2; stride > 32; stride /= 2)
	{
		if (tid < stride && tid + stride < l / 2)
		{
			sdata[tid] = min(sdata[tid], sdata[tid + stride]);
		}
		__syncthreads();
	}

	if (tid < 32)
	{
		warpReduce(sdata, tid, l);
	}

	if (tid == 0)
	{
		// printf("%d: %f\n", blockIdx.x, sdata[0]);
		d_Pba[blockIdx.x] = sdata[0];
	}
}

__device__ void warpReduce(volatile float *sdata, unsigned int tid, int l)
{
	sdata[tid] = (tid + 32 < l / 2) ? min(sdata[tid], sdata[tid + 32]) : sdata[tid];
	sdata[tid] = (tid + 16 < l / 2) ? min(sdata[tid], sdata[tid + 16]) : sdata[tid];
	sdata[tid] = min(sdata[tid], sdata[tid + 8]);
	sdata[tid] = min(sdata[tid], sdata[tid + 4]);
	sdata[tid] = min(sdata[tid], sdata[tid + 2]);
	sdata[tid] = min(sdata[tid], sdata[tid + 1]);
}