//
// Created by herve on 13-04-2024.
//

#include "histogram_eq_par.h"
#include <omp.h>

namespace cp_par {
    constexpr auto HISTOGRAM_LENGTH = 256;

    static float inline prob(const int x, const int size) {
        return (float) x / (float) size;
    }

    static unsigned char inline clamp(unsigned char x) {
        return std::min(std::max(x, static_cast<unsigned char>(0)), static_cast<unsigned char>(255));
    }

    static unsigned char inline correct_color(float cdf_val, float cdf_min) {
        return clamp(static_cast<unsigned char>(255 * (cdf_val - cdf_min) / (1 - cdf_min)));
    }

    static void histogram_equalization_par(const int width, const int height,
                                       const float *input_image_data,
                                       float *output_image_data,
                                       const std::shared_ptr<unsigned char[]> &uchar_image,
                                       const std::shared_ptr<unsigned char[]> &gray_image,
                                       int (&histogram)[HISTOGRAM_LENGTH],
                                       float (&cdf)[HISTOGRAM_LENGTH]) {

        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        #pragma omp parallel for
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = (unsigned char) (255 * input_image_data[i]);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++) {
                auto idx = i * width + j;
                auto r = uchar_image[3 * idx];
                auto g = uchar_image[3 * idx + 1];
                auto b = uchar_image[3 * idx + 2];
                gray_image[idx] = static_cast<unsigned char>(0.21 * r + 0.71 * g + 0.07 * b);
            }

        std::fill(histogram, histogram + HISTOGRAM_LENGTH, 0);
        #pragma omp parallel for
        for (int i = 0; i < size; i++)
            #pragma omp atomic
            histogram[gray_image[i]]++;

        cdf[0] = prob(histogram[0], size);
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf[i] = cdf[i - 1] + prob(histogram[i], size);

        auto cdf_min = cdf[0];
        for (int i = 1; i < HISTOGRAM_LENGTH; i++)
            cdf_min = std::min(cdf_min, cdf[i]);

        #pragma omp parallel for
        for (int i = 0; i < size_channels; i++)
            uchar_image[i] = correct_color(cdf[uchar_image[i]], cdf_min);

        #pragma omp parallel for
        for (int i = 0; i < size_channels; i++)
            output_image_data[i] = static_cast<float>(uchar_image[i]) / 255.0f;
    }

    wbImage_t iterative_histogram_equalization_par(wbImage_t &input_image, int iterations) {

        const auto width = wbImage_getWidth(input_image);
        const auto height = wbImage_getHeight(input_image);
        constexpr auto channels = 3;
        const auto size = width * height;
        const auto size_channels = size * channels;

        wbImage_t output_image = wbImage_new(width, height, channels);
        float *input_image_data = wbImage_getData(input_image);
        float *output_image_data = wbImage_getData(output_image);

        std::shared_ptr<unsigned char[]> uchar_image(new unsigned char[size_channels]);
        std::shared_ptr<unsigned char[]> gray_image(new unsigned char[size]);

        int histogram[HISTOGRAM_LENGTH];
        float cdf[HISTOGRAM_LENGTH];

        // o shared talvez faça sentido... (ver depois)
        #pragma omp parallel for shared(width,height,input_image_data,output_image_data, uchar_image, gray_image,histogram, cdf)
        for (int i = 0; i < iterations; i++) {
            histogram_equalization_par(width, height,
                                   input_image_data, output_image_data,
                                   uchar_image, gray_image,
                                   histogram, cdf);
            // para ter a certeza que a proxima iteracao usa o resultado da anterior
            #pragma omp barrier
            #pragma omp single
            {
                input_image_data = output_image_data;
            }
        }

        return output_image;
    }
}
