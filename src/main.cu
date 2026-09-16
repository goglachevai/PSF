#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <filesystem>

#include "distance.cuh"
#include "mpdist.cuh"
#include "snippets.cuh"

#include <chrono>

namespace fs = std::filesystem;

void start_timer(cudaEvent_t *start)
{
    cudaEventCreate(start);
    cudaEventRecord(*start);
}

float stop_timer(cudaEvent_t *start, cudaEvent_t *stop)
{
    cudaEventCreate(stop);
    cudaEventRecord(*stop);
    cudaEventSynchronize(*stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, *start, *stop);
    cudaEventDestroy(*start);
    cudaEventDestroy(*stop);
    return milliseconds;
}

void device_to_file(float *d_array, int numCols, int numRows, std::string filename)
{
    std::ofstream output(filename);
    if (!output.is_open())
    {
        std::cerr << "Error: Could not open output file " << filename << std::endl;
        return;
    }

    float *h_array = new float[numCols * numRows];
    cudaMemcpy(h_array, d_array, numCols * numRows * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            output << h_array[i * numCols + j] << " ";
        }
        output << std::endl;
    }
    delete[] h_array;
    output.close();
}

int nextPowerOf2(int n)
{
    int power = 1;
    while (power < n)
    {
        power *= 2;
    }
    return power;
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <time_series_length> <segment_length> <num_snippets>" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputDir = argv[2];
    int timeSeriesLength = std::stoi(argv[3]);
    int segmentLength = std::stoi(argv[4]);
    int numSnippets = std::stoi(argv[5]);

    int subsequenceLength = segmentLength / 2;

    if (segmentLength <= 0 || numSnippets <= 0 || timeSeriesLength <= 0)
    {
        std::cerr << "Error: segment_length, num_snippets, and time_series_length must be positive integers." << std::endl;
        return 1;
    }

    std::ifstream input(inputFile);
    if (!input.is_open())
    {
        std::cerr << "Error: Could not open input file " << inputFile << std::endl;
        return 1;
    }

    try
    {
        fs::create_directories(outputDir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: Could not create output directory " << outputDir << ": " << e.what() << std::endl;
        return 1;
    }

    fs::path indiciesFilePath = fs::path(outputDir) / "indicies.txt";
    fs::path snippetsFilePath = fs::path(outputDir) / "snippets.txt";
    fs::path fracsFilePath = fs::path(outputDir) / "fracs.txt";
    fs::path labelsFilePath = fs::path(outputDir) / "labels.txt";
    fs::path profilesFilePath = fs::path(outputDir) / "profiles.txt";

    std::ofstream indiciesFile(indiciesFilePath.string());
    if (!indiciesFile.is_open())
    {
        std::cerr << "Error: Could not open indicies file " << indiciesFilePath << std::endl;
        return 1;
    }

    std::ofstream snippetsFile(snippetsFilePath.string());
    if (!snippetsFile.is_open())
    {
        std::cerr << "Error: Could not open snippets file " << snippetsFilePath << std::endl;
        return 1;
    }

    std::ofstream fracsFile(fracsFilePath.string());
    if (!fracsFile.is_open())
    {
        std::cerr << "Error: Could not open fracs file " << fracsFilePath << std::endl;
        return 1;
    }

    std::ofstream labelsFile(labelsFilePath.string());
    if (!labelsFile.is_open())
    {
        std::cerr << "Error: Could not open labels file " << labelsFilePath << std::endl;
        return 1;
    }

    std::ofstream profilesFile(profilesFilePath.string());
    if (!profilesFile.is_open())
    {
        std::cerr << "Error: Could not open profiles file " << profilesFilePath << std::endl;
        return 1;
    }

    std::ofstream logFile("log.txt", std::ios::app);
    if (!logFile.is_open())
    {
        std::cerr << "Error: Could not open log file log.txt" << std::endl;
        return 1;
    }

    float *timeSeriesData = new float[timeSeriesLength];
    int index = 0;

    std::string line;
    while (std::getline(input, line) && index < timeSeriesLength)
    {
        std::istringstream iss(line);
        float value;
        while (iss >> value && index < timeSeriesLength)
        {
            timeSeriesData[index++] = value;
        }
    }

    int numSegments = timeSeriesLength / segmentLength;
    int *segmentIndices = new int[numSegments];
    for (int i = 0; i < numSegments; ++i)
    {
        segmentIndices[i] = i * segmentLength;
    }

    float *d_timeSeriesData;
    cudaMalloc(&d_timeSeriesData, timeSeriesLength * sizeof(float));
    cudaMemcpy(d_timeSeriesData, timeSeriesData, timeSeriesLength * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float total_time = 0;

    int numRows = segmentLength - subsequenceLength + 1;
    int numCols = timeSeriesLength - subsequenceLength + 1;
    int profileLength = numCols - numRows;

    float *d_mpdistProfiles;
    cudaMalloc(&d_mpdistProfiles, numSegments * profileLength * sizeof(float));
    float *h_mpdistProfiles = new float[numSegments * profileLength];

    for (int i = 0; i < numSegments; i++)
    {
        int targetIndex = segmentIndices[i];

        float *d_distances;
        cudaMalloc(&d_distances, numRows * numCols * sizeof(float));
        start_timer(&start);
        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid((numRows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                           (numCols + threadsPerBlock.y - 1) / threadsPerBlock.y);
        calculateEuclideanDistance<<<blocksPerGrid, threadsPerBlock>>>(d_timeSeriesData + targetIndex, segmentLength, d_timeSeriesData, timeSeriesLength, subsequenceLength, d_distances);
        total_time += stop_timer(&start, &stop);

        float *d_allPab;
        cudaMalloc(&d_allPab, numRows * numCols * sizeof(float));

        start_timer(&start);
        dim3 numBlocks((numCols + 255) / 256, numRows);
        calculateRowMinimums<<<numBlocks, 256>>>(d_distances, numRows, numCols, d_allPab);
        total_time += stop_timer(&start, &stop);

        float *d_allPba;
        cudaMalloc(&d_allPba, numCols * sizeof(float));
        start_timer(&start);
        calculateColumnMinimums<<<(numCols + 255) / 256, 256>>>(d_distances, numRows, numCols, d_allPba);
        total_time += stop_timer(&start, &stop);

        float *d_allPabba;
        cudaMalloc(&d_allPabba, 2 * numRows * (numCols - numRows) * sizeof(float));
        createPabba<<<(numCols - numRows + 255) / 256, 256>>>(d_allPab, d_allPba, numRows, numCols, d_allPabba);

        float *d_mpdistProfile = d_mpdistProfiles + profileLength * i;

        int k = (int)(0.05 * (2 * numRows));
        if (k < 1)
            k = 1;
        if (k > 2 * numRows)
            k = 2 * numRows;
        start_timer(&start);
        size_t sharedMem = sizeof(float) * 2 * numRows;
        calculateMPDistProfile<<<numCols - numRows, 256, sharedMem>>>(numCols - numRows, 2 * numRows, k, d_allPabba, d_mpdistProfile);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(err));
        }
        total_time += stop_timer(&start, &stop);

        cudaFree(d_allPabba);
        cudaFree(d_allPab);
        cudaFree(d_allPba);
        cudaFree(d_distances);

        cudaMemcpy(h_mpdistProfiles + i * profileLength, d_mpdistProfile, profileLength * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_mpdistProfiles);

    SnippetSelection selection = selectSnippets(h_mpdistProfiles, numSegments, profileLength, numSnippets);

    for (int i = 0; i < numSnippets; i++)
    {
        int snippet = selection.snippets[i];

        indiciesFile << snippet * segmentLength << " ";

        fracsFile << selection.fracs[i] << " ";

        for (int j = 0; j < profileLength; j++)
        {
            profilesFile << h_mpdistProfiles[snippet * profileLength + j] << " ";
        }
        profilesFile << std::endl;

        for (int j = 0; j < segmentLength; j++)
        {
            snippetsFile << timeSeriesData[snippet * segmentLength + j] << " ";
        }
        snippetsFile << std::endl;
    }

    for (int j = 0; j < profileLength; j++)
    {
        labelsFile << selection.labels[j] << " ";
    }
    labelsFile << std::endl;

    logFile << inputFile << " " << total_time << std::endl;

    delete[] h_mpdistProfiles;

    std::cout << "Total runtime: " << total_time << " ms" << std::endl;

    delete[] timeSeriesData;
    delete[] segmentIndices;

    // Close files
    input.close();
    logFile.close();
    indiciesFile.close();
    snippetsFile.close();
    fracsFile.close();
    labelsFile.close();
    profilesFile.close();

    return 0;
}