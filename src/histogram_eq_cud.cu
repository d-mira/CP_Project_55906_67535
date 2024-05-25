#include "histogram_eq_par.h"
#include <cub/cu.cuh>
#include <wb.h>

namespace cp_cub {
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

    //TODO
  } 
     
  
} 
