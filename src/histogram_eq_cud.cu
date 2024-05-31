#include "histogram_eq.h"
#include <cub/cub.cuh>
#include <wb.h>

#define THREADS_PER_BLOCK 512
//Extra threads in case the image size is not divisible into numBlocks
#define THREADS_PER_BLOCK1 511


namespace cp::cub {

    constexpr auto HISTOGRAM_LENGTH = 256;

    static float inline calc_prob(const int x, const int size) {
        return (float) x / (float) size;
    }

    static unsigned char inline clamp(unsigned char x) {
        return std::min(std::max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
    }

    static unsigned char inline correct_color(float cdf_val, float cdf_min) {
        return clamp(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
    }

    __global__ void convert_to_uchar(const float* input_image_data, unsigned char* uchar_image, int size_channels) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size_channels) {
            uchar_image[idx] = static_cast<unsigned char>(255 * input_image_data[idx]);
        }
    }
    __global__ void compute_gray_image(const unsigned char* uchar_image, unsigned char* gray_image, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            int r = uchar_image[3 * idx];
            int g = uchar_image[3 * idx + 1];
            int b = uchar_image[3 * idx + 2];
            gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
        }
    }
    __global__ void save_to_output(const unsigned char* uchar_image, float* output_image_data, int size_channels) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size_channels) {
            output_image_data[idx] = static_cast<float>(uchar_image[idx]) / 255.0f;
        }
    }

    __global__ void build_histogram(int* histogram, const unsigned char* gray_image, int size_channels) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        //Race conditions might apply, so maybe need to take care of that here

        if(idx < size_channels) {
            //histogram[gray_image[idx]]++;
            atomicAdd(&histogram[gray_image[idx]], 1)
        }

        __syncthreads();
    }

    //Pre-calculate an array of prob values, to improve runtime of cdf calculation
    __global__ void calc_prob_array(const int* histogram, int size, float* prob) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx < HISTOGRAM_LENGTH) {
            prob[idx] = calc_prob(histogram[idx], size);
        }
    }

    //Function calculated iteratively, by a single GPU thread
    __global__ void cdf_calculation(float* cdf, const float* prob) {
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob[i];
    }

    //Function calculated iteratively, by a single GPU thread
    __global void cdf_min_calc(float* cdf, float* cdf_min) {
        for(int i = 0; i < HISTOGRAM_LENGTH; i++) {
            cdf_min = std::min(cdf_min, cdf[i]);
        }
    }

    __global__ void correct_img_color(int size_channels, float* cdf, float cdf_min, unsigned char* uchar_image) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if(idx < size_channels) {
            uchar_image[idx] = correct_color(cdf[uchar_image[idx]], cdf_min);
        }
    }

    //TODO - completar histogram_equalization


      void histogram_equalization_cub(const int width, const int height,
                                               const float *input_image_data,
                                               float *output_image_data,
                                               const std::shared_ptr<unsigned char[]> &uchar_image,
                                               const std::shared_ptr<unsigned char[]> &gray_image,
                                               int (&histogram)[HISTOGRAM_LENGTH],
                                               float (&cdf)[HISTOGRAM_LENGTH]) {
      /*void histogram_equalization_cub(const int width, const int height,
                                      const float *input_image_data,
                                      float *output_image_data,
                                      unsigned char* uchar_image,
                                      unsigned char* gray_image,
                                      int (&histogram)[HISTOGRAM_LENGTH],
                                      float (&cdf)[HISTOGRAM_LENGTH]){*/

        int size = width * height;
        int size_channels = size * 3;

        float* prob;
        float cdf_min;

        int d_width, d_height, d_size, d_size_channels;

        float* d_input_image_data;
        float* d_output_image_data;

        unsigned char* d_uchar_image;
        unsigned char* d_gray_image;

        int* d_histogram;
        float* d_cdf;

        cudaMalloc((void **)&prob, HISTOGRAM_LENGTH*sizeof(float));
        cudaMalloc((void *) &cdf_min, sizeof(float));

        cudaMalloc((void *)&d_width, sizeof(int));
        cudaMalloc((void *)&d_height, sizeof(int));
        cudaMalloc((void *)&d_size, sizeof(int));
        cudaMalloc((void *)&d_size_channels, sizeof(int));
        cudaMalloc((void **)&d_input_image_data, size_channels*sizeof(float));
        cudaMalloc((void **)&d_output_image_data, size_channels*sizeof(float));
        cudaMalloc((void **)&d_uchar_image, size_channels*sizeof(unsigned char));
        cudaMalloc((void **)&d_gray_image, size*sizeof(unsigned char));
        cudaMalloc((void **)&d_histogram, HISTOGRAM_LENGTH*sizeof(int));
        cudaMalloc((void **)&d_cdf, HISTOGRAM_LENGTH*sizeof(float));

        cudaMemcpy(d_width, width, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_height, height, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_size, size, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_size_channels, size_channels, sizeof(int), cudaMemcpyHostToDevice);

        cudaMemcpy(d_input_image_data, input_image_data, size_channels*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output_image_data, output_image_data, size_channels*sizeof(float), cudaMemcpyHostToDevice);

        int numBlocks = (size_channels + THREADS_PER_BLOCK1)/ THREADS_PER_BLOCK;

        //TODO - Ensure the whole image gets processed (what happens when we can't allocate enough thread blocks to cover the image)?
        convert_to_uchar<<<numBlocks, THREADS_PER_BLOCK>>>(d_input_image_data, d_uchar_image, d_size_channels);

        compute_gray_image<<<numBlocks, THREADS_PER_BLOCK>>>(d_uchar_image, d_gray_image, size);

        std::fill(d_histogram, d_histogram+HISTOGRAM_LENGTH, 0);
        build_histogram<<<numBlocks, THREADS_PER_BLOCK>>>(d_histogram, d_gray_image, d_size_channels);

        calc_prob_array<<<numBlocks, THREADS_PER_BLOCK>>>(d_histogram, d_size, prob);
        cdf_calculation<<<1, 1>>>(d_cdf, prob);
        cdf_min_calc<<<1,1>>>(d_cdf, &cdf_min);

        correct_img_color<<<numBlocks, THREADS_PER_BLOCK>>>(d_size_channels, d_cdf, cdf_min, d_uchar_image);

        save_to_output<<<numBlocks, THREADS_PER_BLOCK>>>(d_uchar_image, d_output_image_data, d_size_channels);

        cudaMemcpy(output_image_data, d_output_image_data, size_channels*sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_width);
        cudaFree(d_height);
        cudaFree(d_size);
        cudaFree(d_size_channels);
        cudaFree(d_input_image_data);
        cudaFree(d_output_image_data);
        cudaFree(d_uchar_image);
        cudaFree(d_gray_image);
        cudaFree(d_histogram);
        cudaFree(d_cdf);

        cudaFree(prob);
        cudaFree(cdf_min);
      }

      wbImage_t iterative_histogram_equalization_cub(wbImage_t &input_image, int iterations){
          const int width = wbImage_getWidth(input_image);
          const int height = wbImage_getHeight(input_image);
          const int size = width * height;
          const int size_channels = size * 3;

          //int d_width, d_height, d_size, d_size_channels;

          wbImage_t output_image = wbImage_new(width, height, 3);
          float* input_image_data = wbImage_getData(input_image);
          float* output_image_data = wbImage_getData(output_image);

          //float* d_input_image_data;
          //float* d_output_image_data;

          std::shared_ptr<unsigned char[]> uchar_image(new unsigned char[size_channels]);
          std::shared_ptr<unsigned char[]> gray_image(new unsigned char[size]);

          /*unsigned char* d_uchar_image;
          unsigned char* d_gray_image;*/

          int histogram[HISTOGRAM_LENGTH];
          float cdf[HISTOGRAM_LENGTH];

         // int* d_histogram;
         // float* d_cdf;

          /*cudaMalloc((void *)&d_width, sizeof(int));
          cudaMalloc((void *)&d_height, sizeof(int));
          cudaMalloc((void *)&d_size, sizeof(int));
          cudaMalloc((void *)&d_size_channels, sizeof(int));
          cudaMalloc((void **)&d_input_image_data, size_channels*sizeof(float));
          cudaMalloc((void **)&d_output_image_data, size_channels*sizeof(float));
          cudaMalloc((void **)&d_uchar_image, size_channels*sizeof(unsigned char));
          cudaMalloc((void **)&d_gray_image, size*sizeof(unsigned char));
          cudaMalloc((void **)&d_histogram, HISTOGRAM_LENGTH*sizeof(int));
          cudaMalloc((void **)&d_cdf, HISTOGRAM_LENGTH*sizeof(float));

          cudaMemcpy(d_width, width, sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(d_height, height, sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(d_size, size, sizeof(int), cudaMemcpyHostToDevice);
          cudaMemcpy(d_size_channels, size_channels, sizeof(int), cudaMemcpyHostToDevice);

          cudaMemcpy(d_input_image_data, input_image_data, size_channels*sizeof(float), cudaMemcpyHostToDevice);
          cudaMemcpy(d_output_image_data, output_image_data, size_channels*sizeof(float), cudaMemcpyHostToDevice);*/

          for (int i = 0; i < iterations; i++) {
              histogram_equalization_cub(width, height, input_image_data, output_image_data,
                                         uchar_image, gray_image, histogram, cdf);
              input_image_data = output_image_data;
              //d_input_image_data = d_output_image_data;
          }

          /*cudaMemcpy(output_image_data, d_output_image_data, size_channels*sizeof(float), cudaMemcpyDeviceToHost);

          cudaFree(d_width);
          cudaFree(d_height);
          cudaFree(d_size);
          cudaFree(d_size_channels);
          cudaFree(d_input_image_data);
          cudaFree(d_output_image_data);
          cudaFree(d_uchar_image);
          cudaFree(d_gray_image);
          cudaFree(d_histogram);
          cudaFree(d_cdf);*/

          return output_image;
      }
  } 
     
  
} 
