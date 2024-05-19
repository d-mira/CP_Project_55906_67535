//
// Created by Diogo Pinto on 19/05/2024.
//

#ifndef PROJECT_HISTOGRAM_EQ_PAR_H
#define PROJECT_HISTOGRAM_EQ_PAR_H


#include "wb.h"

namespace cp {
    wbImage_t iterative_histogram_equalization_par(wbImage_t &input_image, int iterations = 1);
}


#endif //PROJECT_HISTOGRAM_EQ_PAR_H
