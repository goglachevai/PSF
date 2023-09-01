/*+++++++++++++++++++++++++++++++++
Project: PSF (Parallel Snippet-Finder)
Source file: main.cu
Purpose: Parallel implementation of the PSF algorithm in CUDA
Author(s): Andrey Goglachev (goglachevai@susu.ru)
+++++++++++++++++++++++++++++++++*/

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "SCAMP.h"
#include "common.h"
#include "scamp_exception.h"
#include "scamp_utils.h"
#include <iostream>
#include <fstream>
#include <math.h>
#include "MPdist.h"
#include "SnippetFinder.h"
#include <chrono>

int main(int argc, char **argv)
{
    char *file_name = argv[1];
    unsigned int n = atoi(argv[2]);
    unsigned int m = atoi(argv[3]);
    char *output_file_name = argv[4];

    unsigned int l = m / 2;

    bool self_join, computing_rows, computing_cols;
    size_t start_row = 0;
    size_t start_col = 0;

    std::vector<double> Ta_h(n);

    std::ifstream is(file_name);
    for (int i = 0; i < n; i++)
    {
        is >> Ta_h[i];
    }

    int n_x = Ta_h.size() - l + 1;
    int n_y = n_x;

    if (n_x < 1 || n_y < 1)
    {
        printf("Error: window size must be smaller than the timeseries length\n");
        return 1;
    }
    int N = (n + m - 1) / m - 1;
    unsigned __int64 size = (n - l + 1) * (m - l + 1) * sizeof(float);
    float *d_profiles;
    cudaMalloc(&d_profiles, N * (n - m) * sizeof(float));
    SCAMP::SCAMPArgs args;
    args.window = l;
    args.has_b = true;
    args.profile_a.type = ParseProfileType("1NN_INDEX");
    args.profile_b.type = ParseProfileType("1NN_INDEX");
    args.precision_type = GetPrecisionType(false, true, false, false);
    args.profile_type = ParseProfileType("1NN_INDEX");
    args.timeseries_a = Ta_h;
    args.silent_mode = true;
    cudaError_t code = cudaMalloc(&args.distance_matrix, size);
    if (code != cudaSuccess)
    {
        printf("Memory error");
    }

    for (int i = 0; i < N; i++)
    {
        printf("%d: %d\n", i, N);
        auto first = Ta_h.cbegin() + i * m;
        auto last = Ta_h.cbegin() + (i * m + m);
        std::vector<double> Tb_h(first, last);
        args.timeseries_b = std::move(Tb_h);
        try
        {
            InitProfileMemory(&args);
            SCAMP::do_SCAMP(&args);
        }
        catch (const SCAMPException &e)
        {
            std::cout << e.what() << "\n";
            exit(1);
        }
        cudaDeviceSynchronize();
        MPdist_(args.distance_matrix, d_profiles, n, m, l, i);
        cudaDeviceSynchronize();
    }
    snippet_finder(d_profiles, n, m, 2);

    std::ifstream os(output_file_name);
    for (Snippet &it : snippets)
    {
        printf("idx: %d, frac: %f\n", it.index, it.frac);
        os << it.index << endl
           << it.frac << endl;
    }
    os.close();
    cudaFree(d_profiles);

    return 0;
}
