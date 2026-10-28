#pragma once
#include <vector>

struct SnippetSelection
{
    std::vector<int> snippets;
    std::vector<int> labels;
    std::vector<float> fracs;
};

SnippetSelection selectSnippets(const float *mpdistProfiles, int numSegments, int profileLength, int numSnippets);
