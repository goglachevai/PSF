#pragma once
#include <vector>

struct SnippetSelection
{
    std::vector<int> snippets; // size numSnippets: chosen segment index per round
    std::vector<int> labels;   // size profileLength: nearest-snippet round-index per profile position
    std::vector<float> fracs;  // size numSnippets: fraction of profile positions labeled to each snippet
};

SnippetSelection selectSnippets(const float *mpdistProfiles, int numSegments, int profileLength, int numSnippets);
