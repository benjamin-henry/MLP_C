#ifndef LOSSES_H
#define LOSSES_H

#include "math.h"

#ifdef __cplusplus
extern "C" {
#endif


#include "Utils.h"

enum Loss {
  Categorical_Crossentropy = 0,
  Binary_Crossentropy,
  MSE,
};

enum Reduction {
  NONE = 0,
  SUM,
  SUM_OVER_BATCH_SIZE
};


// double * MSE_LOSS_NONE(unsigned int batch_size, unsigned int output_shape, double* y_true[], double* y_pred[]);
// double MSE_LOSS_SUM(unsigned int batch_size, unsigned int output_shape, double* y_true[], double* y_pred[]);
// double MSE_LOSS_SUM_OVER_BATCH_SIZE(unsigned int batch_size, unsigned int output_shape, double* y_true[], double* y_pred[]);


#ifdef __cplusplus
}
#endif


#endif
