#include "snippets.cuh"
#include <limits>

SnippetSelection selectSnippets(const float* mpdistProfiles, int numSegments, int profileLength, int numSnippets)
{
    SnippetSelection result;
    result.snippets.resize(numSnippets);
    result.labels.assign(profileLength, 0);
    result.fracs.resize(numSnippets);

    std::vector<float> curve(profileLength, std::numeric_limits<float>::infinity());
    float min_curve_s = std::numeric_limits<float>::infinity();

    for (int snippet = 0; snippet < numSnippets; snippet++)
    {
        int snippet_candidate = 0;
        for (int seg = 0; seg < numSegments; seg++)
        {
            const float* profile = mpdistProfiles + (size_t)profileLength * seg;
            std::vector<float> temp_curve(profileLength);
            for (int j = 0; j < profileLength; j++)
            {
                if (profile[j] < curve[j])
                {
                    temp_curve[j] = profile[j];
                }
                else
                {
                    temp_curve[j] = curve[j];
                }
            }
            float temp_curve_s = 0.0f;
            for (int j = 0; j < profileLength; j++)
            {
                temp_curve_s += temp_curve[j];
            }
            if (temp_curve_s < min_curve_s)
            {
                min_curve_s = temp_curve_s;
                snippet_candidate = seg;
            }
        }

        const float* profile = mpdistProfiles + (size_t)profileLength * snippet_candidate;
        for (int j = 0; j < profileLength; j++)
        {
            if (profile[j] < curve[j])
            {
                curve[j] = profile[j];
                result.labels[j] = snippet;
            }
        }
        result.snippets[snippet] = snippet_candidate;
    }

    for (int snippet = 0; snippet < numSnippets; snippet++)
    {
        float count = 0;
        for (int j = 0; j < profileLength; j++)
        {
            if (result.labels[j] == snippet)
            {
                count++;
            }
        }
        result.fracs[snippet] = count / profileLength;
    }

    return result;
}
