#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include "SnippetFinder.h"
#include <vector>

std::vector<Snippet> snippet_finder(float* ts, int n, int m, int K) {
	std::vector<Snippet> snippets(K);
	int N = (n + m - 1) / m - 1;
	int numThreads = 256;
	int numBlocks = n / 256 / 2 + 1;
	float* h_M = (float*)malloc((n - m) * sizeof(float));
	for (int i = 0; i < (n - m); i++) {
		h_M[i] = std::numeric_limits<float>::max();
	}
	float* d_M;
	cudaMalloc(&d_M, (n - m) * sizeof(float));
	cudaMemcpy(d_M, h_M, (n - m) * sizeof(float), cudaMemcpyHostToDevice);
	float* h_profile_area = (float*)malloc(N * (n - m) * sizeof(float));
	float* d_profile_area = (float*)malloc(numBlocks * sizeof(float));
	cudaMalloc(&d_profile_area, numBlocks * sizeof(float));
	float sum = 0, min_sum = std::numeric_limits<float>::max();
	int* h_neighbours = (int*)malloc((n - m) * sizeof(int));
	int* d_neighbours;
	cudaMalloc(&d_neighbours, (n - m) * sizeof(int));
	int min_idx = 0;

	for (int c = 0; c < K; c++) {
		
		float minArea = std::numeric_limits<float>::infinity();

		for (int i = 0; i < N; i++) {
			sum = 0;
			get_profile_area << <numBlocks, numThreads, numThreads * sizeof(float) >> > (d_M, ts + (n - m) * i, d_profile_area, n - m);
			cudaMemcpy(h_profile_area, d_profile_area, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
			for (int j = 0; j < numBlocks; j++)
			{
				sum += h_profile_area[j];
				//printf("%d, %f\n", j, h_profile_area[j]);
			}
			//printf("%d, %f\n", i, sum);
			if (sum < min_sum) {
				min_sum = sum;
				min_idx = i;
			}
		}
		set_min<<<numBlocks * 2, numThreads>>>(d_M, ts, d_neighbours, min_idx, n - m);
		//printf("Min: %d, %f\n", min_idx, min_sum);
		snippets[c] = Snippet();
		snippets[c].index = min_idx;
	}
	cudaMemcpy(h_neighbours, d_neighbours, (n - m) * sizeof(int), cudaMemcpyDeviceToHost);
	for (Snippet& it : snippets) {
		int a = 0;
		for (int i = 0; i < n - m; i++) {
			if (h_neighbours[i] == it.index) a++;
		}
		it.frac = static_cast<float>(a) / (n - m);
	}

	return snippets;
}

__global__ void get_profile_area(float* g_M, float* g_D, float* g_profile_area, int n) {
    extern __shared__ int s_data[];
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	float s_1 = 0, s_2 = 0;
	if (g_M[i] > g_D[i]) s_1 = g_D[i]; else s_1 = g_M[i];
	s_data[threadIdx.x] = s_1;
	if (idx + blockDim.x < n) {
		if (g_M[i + blockDim.x] > g_D[i + blockDim.x])
			s_2 = g_D[i + blockDim.x]; else s_2 = g_M[i + blockDim.x];
		s_data[threadIdx.x] = s_1 + s_2;
	}
	__syncthreads();

	for (int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (threadIdx.x < s) {
			s_data[threadIdx.x] += s_data[threadIdx.x + s];
		}
		__syncthreads();
	}

	if (threadIdx.x < 32) {
		warpReduce(s_data, threadIdx.x);
	}

	if (threadIdx.x == 0) {
		g_profile_area[blockIdx.x] = s_data[0];
	}
}

__device__ void warpReduce(volatile int* s_data, int t) {
	s_data[t] += s_data[t + 32];
	s_data[t] += s_data[t + 16];
	s_data[t] += s_data[t + 8];
	s_data[t] += s_data[t + 4];
	s_data[t] += s_data[t + 2];
	s_data[t] += s_data[t + 1];
}

__global__ void set_min(float* g_M, float* g_profiles, int* d_neighbours, int profile_idx, int n) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n && g_M[idx] > g_profiles[profile_idx * n + idx]) {
		g_M[idx] = g_profiles[idx];
		d_neighbours[idx] = profile_idx;
	}
}
/*
__global__ void compute_frac(int* d_neighbours, , int n) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < n && g_M[idx] > g_profiles[profile_idx * n + idx]) {
		g_M[idx] = g_profiles[idx];
		d_neighbours[idx] = profile_idx;
	}
}
*/