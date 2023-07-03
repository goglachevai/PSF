#pragma once
#include <vector>

struct Snippet {
	int index;
	float frac;
};

std::vector<Snippet> snippet_finder(float* ts, int n, int m, int K);

__global__ void get_profile_area(float* g_M, float* g_D, float* g_profile_area, int n);

__device__ void warpReduce(volatile int* s_data, int t);

__global__ void set_min(float* g_M, float* g_profiles, int* d_neighbours, int profile_idx, int n);


