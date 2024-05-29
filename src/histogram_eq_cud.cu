#include "histogram_eq_par.h"
#include <cub/cu.cuh>
#include <wb.h>

namespace cp::cub {
    constexpr auto HISTOGRAM_LENGTH = 256;

    static float inline prob(const int x, const int size) {
        return (float) x / (float) size;
    }

    __global__ void convert_to_uchar(const float* input_image_data, unsigned char* uchar_image, int size_channels) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size_channels) {
            uchar_image[idx] = static_cast<unsigned char>(255 * input_image_data[idx]);
        }
    }
    __global__ void compute_gray_image(const unsigned char* uchar_image, unsigned char* gray_image, int width, int height) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < width * height) {
            int r = uchar_image[3 * idx];
            int g = uchar_image[3 * idx + 1];
            int b = uchar_image[3 * idx + 2];
            gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
        }
    }
    __global__ void uchar_to_float(const unsigned char* uchar_image, float* output_image_data, int size_channels) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size_channels) {
            output_image_data[idx] = static_cast<float>(uchar_image[idx]) / 255.0f;
        }
    }

    __global__ void build_histogram(int* histogram, unsigned char* gray_image) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        std::fill(histogram, histogram+HISTOGRAM_LENGTH, 0);

        //Race conditions might apply, so maybe need to take care of that here

        if(idx < HISTOGRAM_LENGTH) {
            histogram[gray_image[idx]]++;
        }
    }

    //Pre-calculate an array of prob values, to improve runtime of cdf calculation
    __global__ void calc_prob_array() {

    }

    //Function calculated iteratively, by a single GPU thread
    __global__ void cdf_calculation(int* histogram, int size, float* cdf, float* prob) {
        cdf[0] = prob[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob[0];
    }

    __global__ void min_cdf(float* cdf, float* cdf_min) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockIdx.x * gridDim.x;

        int numThreads = gridDim.x/blockDim.x;
        float *localMins = new float[numThreads];

        //Each thread will calculate a min value of the values it
        // got attributed to ("localMin")
        for(int i = idx; i < HISTOGRAM_LENGTH; i += stride) {
            localMins[idx] = std::min(localMins[idx], cdf[i]);
        }

        __syncThreads();

        //Calculate the total min
        for(int i = 0; i < numThreads; i++) {
            cdf_min = std::min(cdf_min, localMins[i]);
        }

    }

    //TODO - adicionar restantes cuda kernals histogram and cdf calculation e completar histogram_equalization

    //Should this be "__global__"? if so, do we need to extend the visibility
    // of its arguments to all threads or is this default behaviour?
      void histogram_equalization(const int width, const int height,
                                               const float *input_image_data,
                                               float *output_image_data,
                                               const std::shared_ptr<unsigned char[]> &uchar_image,
                                               const std::shared_ptr<unsigned char[]> &gray_image,
                                               int (&histogram)[HISTOGRAM_LENGTH],
                                               float (&cdf)[HISTOGRAM_LENGTH]) {
      //TODO
      }

      wbImage_t iterative_histogram_equalization_cub(wbImage_t &input_image, int iterations){
          const int width = wbImage_getWidth(input_image);
          const int height = wbImage_getHeight(input_image);
          const int size = width * height;
          const int size_channels = size * 3;

          wbImage_t output_image = wbImage_new(width, height, 3);
          float* input_image_data = wbImage_getData(input_image);
          float* output_image_data = wbImage_getData(output_image);

          for (int i = 0; i < iterations; i++) {
              histogram_equalization_par(width, height, input_image_data, output_image_data);
              input_image_data = output_image_data;
          }

          return output_image;
      }
  } 
     
  
} 
