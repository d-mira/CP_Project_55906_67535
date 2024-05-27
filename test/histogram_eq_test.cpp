#include <filesystem>
#include "gtest/gtest.h"
#include "../include/histogram_eq.h"

using namespace cp;

#define DATASET_FOLDER "../../dataset/"
#define OUTPUT_DATASET_FOLDER "../../outputDataset"

TEST(HistogramEq, Input01_4) {

    wbImage_t inputImage = wbImport(DATASET_FOLDER "input01.ppm");
    wbImage_t outputImage = iterative_histogram_equalization(inputImage, 1);
    // check if the output image is correct

    wbImage_t baseOutput = wbImport(OUTPUT_DATASET_FOLDER "output01.ppm");

    bool result = wbImage_sameQ(outputImage, baseOutput);

    EXPECT_TRUE(result);
}