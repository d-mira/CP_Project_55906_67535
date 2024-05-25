#include "histogram_eq_par.h"
#include <cub/cu.cuh>
#include <wb.h>

namespace cp::cub {
    //TODO - adicionar todas as restantes funções que estão em falta

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
