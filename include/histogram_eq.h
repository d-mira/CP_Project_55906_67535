#ifndef CP_PROJECT_HISTOGRAM_EQ_H
#define CP_PROJECT_HISTOGRAM_EQ_H

#include "wb.h"

namespace cp {
    wbImage_t iterative_histogram_equalization(wbImage_t &input_image, int iterations = 1);
    namespace par {
        wbImage_t iterative_histogram_equalization_par(wbImage_t &input_image, int iterations = 1);
    }
    namespace cub{
        wbImage_t iterative_histogram_equalization_cub(wbImage_t &input_image, int iterations = 1);
    }
}

#endif //CP_PROJECT_HISTOGRAM_EQ_H
