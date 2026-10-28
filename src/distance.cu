#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>

__global__ void calculateEuclideanDistance(const float *d_timeSeries1, int length1, const float *d_timeSeries2, int length2, int subsequenceLength, float *distances)
{
    int idx1 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;

    int rows = length1 - subsequenceLength + 1;
    int cols = length2 - subsequenceLength + 1;
    if (idx1 >= rows || idx2 >= cols)
    {
        return;
    }

    float sumX = 0.0f, sumX2 = 0.0f;
    float sumY = 0.0f, sumY2 = 0.0f;
    float dotXY = 0.0f;
    for (int i = 0; i < subsequenceLength; ++i)
    {
        float x = d_timeSeries1[idx1 + i];
        float y = d_timeSeries2[idx2 + i];
        sumX += x;
        sumX2 += x * x;
        sumY += y;
        sumY2 += y * y;
        dotXY += x * y;
    }

    float m = (float)subsequenceLength;
    float meanX = sumX / m;
    float meanY = sumY / m;
    float varX = sumX2 / m - meanX * meanX;
    float varY = sumY2 / m - meanY * meanY;
    const float eps = 1e-10f;
    float stdX = sqrtf(fmaxf(varX, eps));
    float stdY = sqrtf(fmaxf(varY, eps));

    float covXY = dotXY / m - meanX * meanY;
    float corr = covXY / (stdX * stdY);
    corr = fminf(fmaxf(corr, -1.0f), 1.0f);
    float distSq = 2.0f * m * (1.0f - corr);
    distances[idx1 * cols + idx2] = sqrtf(fmaxf(distSq, 0.0f));
}
