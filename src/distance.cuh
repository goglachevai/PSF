#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void calculateEuclideanDistance(const float* d_timeSeries1, int length1, const float* d_timeSeries2, int length2, int subsequenceLength, float* distances);
