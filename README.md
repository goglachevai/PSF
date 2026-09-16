# PSF (Parallel Snippet Finder)

This repository is related to the PSF (Parallel Snippet-Finder) algorithm that accelerates snippet discovery in time series with a graphics processor. PSF is authored by Andrey Goglachev (goglachevai@susu.ru) and Mikhail Zymbler (mzym@susu.ru), South Ural State University, Chelyabinsk, Russia. The repository contains the PSF's source code (in C, CUDA).

PSF is described in detail in our article Zymbler M., Goglachev A. Fast Summarization of Long Time Series with Graphics Processor // Mathematics. 2022. Vol. 10, No. 10. Article 1781. DOI: [10.3390/math10101781](https://doi.org/10.3390/math10101781)

## Usage

```
PDSS <input_file> <output_dir> <time_series_length> <segment_length> <num_snippets> [candidate_step]
```

| Parameter             | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `input_file`          | Text file containing the time series (numbers separated by spaces/newlines) |
| `output_dir`          | Directory for results (created automatically)                               |
| `time_series_length`  | Length of the input time series                                             |
| `segment_length`      | Length of the snippet                                                       |
| `num_snippets`        | Number of snippets to find                                                  |

Example:

```
PSF GreatBarbet1F_Jog_1800_200.txt results 1800 200 2
```

## Output

The following files are created in `output_dir`:

- `indicies.txt` — the starting index of each found snippet within the original series.
- `snippets.txt` — the values ​​of each snippet.
- `fracs.txt` — the fraction of the series covered by each snippet.
- `labels.txt` — labeling of the input time series.
- `profiles.txt` — the C22dist profile for each found snippet.
