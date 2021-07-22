#ifndef __DENSE_H
#define __DENSE_H

#include "Arduino.h"
#include "Activations.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct dense_t {
  unsigned int _input_shape;
  unsigned int _units;
  unsigned int _activation;
  float** _w; 
  float* _b;
  float* _z;
  float* _output;
} dense_t;

typedef struct dense_t* Dense;

Dense build_dense(unsigned int input_shape, unsigned int units, unsigned int activation);

void randomize_weights(Dense layer, float minimum, float maximum);

void dense_forward(Dense layer, float * input_data);
void dense_activation(Dense layer);
void dense_fSigmoid(Dense layer);
void dense_fTanh(Dense layer);
void dense_fReLU(Dense layer);
void dense_fSoftmax(Dense layer);
void dense_fLinear(Dense layer);


void dense_activation_derivative(Dense layer, float * output);
void dense_fSigmoid_derivative(Dense layer, float * output);
void dense_fTanh_derivative(Dense layer, float * output);
void dense_fReLU_derivative(Dense layer, float * output);
void dense_fSoftmax_derivative(Dense layer, float * output);
void dense_fLinear_derivative(Dense layer, float * output);

#ifdef __cplusplus
}
#endif

#endif
