#include <chrono>
#include <filesystem>
#include "gtest/gtest.h"
#include "../include/histogram_eq.h"

using namespace cp;

#define DATASET_FOLDER "../../dataset/"
#define OUTPUT_DATASET_FOLDER "../../outputDataset/"


struct HistogramEqTestParams {
    std::string inputFileName;
    std::string outputFileName;
    int iterations;
};

class HistogramEqCUDATest : public ::testing::TestWithParam<HistogramEqTestParams> {};

TEST_P(HistogramEqCUDATest, IterativeHistogramEqualizationPar) {
    HistogramEqTestParams params = GetParam();

    GTEST_LOG_(INFO) << "Testing parallelized version for " << params.inputFileName
                     << " with " << params.iterations << " iterations";

    auto start = std::chrono::high_resolution_clock::now();

    wbImage_t inputImage = wbImport((DATASET_FOLDER + params.inputFileName).c_str());
    wbImage_t outputImage = cp::cub::iterative_histogram_equalization_cub(inputImage, params.iterations);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::string baseOutputPath = OUTPUT_DATASET_FOLDER + params.outputFileName;
    wbImage_t baseOutput = wbImport(baseOutputPath.c_str());

    bool result = wbImage_sameQ(outputImage, baseOutput);

    EXPECT_TRUE(result);
    std::cout << "Execution time for " << params.inputFileName << " with " << params.iterations << " iterations: " << elapsed.count() << " milliseconds\n";
    std::cout << "Result: " << std::boolalpha << result << std::endl;
}

INSTANTIATE_TEST_SUITE_P(
        HistogramEqTests,
        HistogramEqCUDATest,
        ::testing::Values(
                HistogramEqTestParams{"input01.ppm", "input01_1.ppm", 1},
                HistogramEqTestParams{"input01.ppm", "input01_100.ppm", 100},
                HistogramEqTestParams{"input01.ppm", "input01_500.ppm", 500},
                HistogramEqTestParams{"input01.pp m", "input01_1000.ppm", 1000},
                HistogramEqTestParams{"borabora_1.ppm", "borabora_1_1.ppm", 1},
                HistogramEqTestParams{"borabora_1.ppm", "borabora_1_100.ppm", 100},
                HistogramEqTestParams{"borabora_1.ppm", "borabora_1_500.ppm", 500},
                HistogramEqTestParams{"borabora_1.ppm", "borabora_1_1000.ppm", 1000},
                HistogramEqTestParams{"sample_5184×3456.ppm", "sample_5184×3456_1.ppm", 1},
                HistogramEqTestParams{"sample_5184×3456.ppm", "sample_5184×3456_100.ppm", 100},
                HistogramEqTestParams{"sample_5184×3456.ppm", "sample_5184×3456_500.ppm", 500},
                HistogramEqTestParams{"sample_5184×3456.ppm", "sample_5184×3456_1000.ppm", 1000}
        )
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
