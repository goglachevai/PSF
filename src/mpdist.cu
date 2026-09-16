#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <float.h>

__global__ void calculateColumnMinimums(
    const float* d_distances,
    int numRows,
    int numCols,
    float* d_allPba)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= numCols) return;

    float minVal = d_distances[col];
    for (int row = 1; row < numRows; ++row) {
        float val = d_distances[row * numCols + col];
        if (val < minVal) minVal = val;
    }
    d_allPba[col] = minVal;
}

__global__ void calculateRowMinimums(
    const float* d_distances,
    int numRows,
    int numCols,
    float* d_allPab)
{
    int row = blockIdx.y;
    int pos_in_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (pos_in_row >= numCols) return;

    int idx = row * numCols + pos_in_row;
    float minVal = d_distances[idx];
    for (int offset = 1; offset < numRows; ++offset) {
		if (pos_in_row + offset >= numCols) break;
        float val = d_distances[idx + offset];
        minVal = fminf(minVal, val);
    }
    d_allPab[idx] = minVal;
}

__global__ void createPabba(
	const float* d_allPab,
	const float* d_allPba,
	int numRows,
	int numCols,
	float* d_allPabba)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numCols - numRows) return;
    for (int row = 0; row < numRows; row++) {
        d_allPabba[idx * 2 * numRows + row] = d_allPba[idx + row];
        d_allPabba[idx * 2 * numRows + numRows + row] = d_allPab[row * numCols + idx];
    }

}

__global__ void calculateMPDistProfile(
    int numRows,
    int numCols,
    int k,
    float* d_allPabba,
    float* d_mpdistProfile)
{
    int row = blockIdx.x;
    if (row >= numRows) return;
    extern __shared__ float shared_row[];
    float* row_data = shared_row;

    for (int col = threadIdx.x; col < numCols; col += blockDim.x) {
        row_data[col] = d_allPabba[row * numCols + col];
    }
    __syncthreads();

    for (int i = 0; i < k; ++i) {
        int local_min_idx = -1;
        float local_min_val = FLT_MAX;
        for (int j = i + threadIdx.x; j < numCols; j += blockDim.x) {
            if (shared_row[j] < local_min_val) {
                local_min_val = shared_row[j];
                local_min_idx = j;
            }
        }

        __shared__ float sh_min_val[1024];
        __shared__ int sh_min_idx[1024];
        sh_min_val[threadIdx.x] = local_min_val;
        sh_min_idx[threadIdx.x] = local_min_idx;
        __syncthreads();

        int tid = threadIdx.x;
        for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (tid < offset) {
                if (sh_min_val[tid + offset] < sh_min_val[tid]) {
                    sh_min_val[tid] = sh_min_val[tid + offset];
                    sh_min_idx[tid] = sh_min_idx[tid + offset];
                }
            }
            __syncthreads();
        }

        int min_idx = sh_min_idx[0];

        if (tid == 0 && min_idx >= 0 && min_idx != i) {
            float tmp = shared_row[i];
            shared_row[i] = shared_row[min_idx];
            shared_row[min_idx] = tmp;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) d_mpdistProfile[row] = row_data[k - 1];
}