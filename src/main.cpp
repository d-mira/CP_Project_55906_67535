#include <filesystem>
#include "histogram_eq.h"
#include <cstdlib>
#include <chrono>
#include <omp.h>
#define OUTPUT_DATASET_FOLDER "../outputDataset/"

int main(int argc, char **argv) {

    #ifdef PROJECT_NAME
        std::cout << "Running executable: " << PROJECT_NAME << std::endl;
    #endif

    if (argc != 4) {
        std::cout << "usage" << argv[0] << " input_image.ppm n_iterations output_image.ppm\n";
        return 1;
    }

    wbImage_t inputImage = wbImport(argv[1]);
    int n_iterations = static_cast<int>(std::strtol(argv[2], nullptr, 10));
    auto start = std::chrono::high_resolution_clock::now();

    #ifdef RUN_SEQUENTIAL
        wbImage_t outputImage_seq = cp::iterative_histogram_equalization(inputImage, n_iterations);

        std::string baseOutputPath = OUTPUT_DATASET_FOLDER + std::string(argv[3]);
        if(std::filesystem::exists(baseOutputPath)) {
            std::cout << "Testing image: " << std::string(argv[1]) << "with "<< std::string(argv[2]) << " iterations\n";
            std::cout << "Base output image: " << baseOutputPath <<"\n";
            wbImage_t baseOutput = wbImport(baseOutputPath.c_str());
            bool result = wbImage_sameQ(outputImage_seq, baseOutput);
            std::cout << "Result: " << result <<" zero=false one=true\n";
        }else {
            wbExport(argv[3], outputImage_seq);
        }
    #endif

    #ifdef RUN_PARALLEL
        wbImage_t outputImage_par = cp::par::iterative_histogram_equalization_par(inputImage, n_iterations);

        std::string baseOutputPath = OUTPUT_DATASET_FOLDER + std::string(argv[3]);
        if(std::filesystem::exists(baseOutputPath)) {
            std::cout << "Testing image: " << std::string(argv[1]) << "with "<< std::string(argv[2]) << " iterations\n";
            std::cout << "Base output image: " << baseOutputPath <<"\n";
            wbImage_t baseOutput = wbImport(baseOutputPath.c_str());
            bool result = wbImage_sameQ(outputImage_par, baseOutput);
            std::cout << "Is Image Equal: " << result <<" zero=false one=true\n";
        }else {
            wbExport(argv[3], outputImage_par);
        }

    #endif

    #ifdef RUN_CUDA
        wbImage_t outputImage_cub = cp::cub::iterative_histogram_equalization_cub(inputImage, n_iterations);
        std::string baseOutputPath = OUTPUT_DATASET_FOLDER + std::string(argv[3]);
        if(std::filesystem::exists(baseOutputPath)) {
            std::cout << "Testing image: " << std::string(argv[1]) << "with "<< std::string(argv[2]) << " iterations\n";
            std::cout << "Base output image: " << baseOutputPath <<"\n";
            wbImage_t baseOutput = wbImport(baseOutputPath.c_str());
            bool result = wbImage_sameQ(outputImage_cub, baseOutput);
            std::cout << "Result: " << result <<" zero=false one=true\n";
        }else {
            wbExport(argv[3], outputImage_cub);
        }
    #endif

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Execution time: " << elapsed.count() << " ms\n";

    return 0;
}
