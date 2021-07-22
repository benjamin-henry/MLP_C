#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "math.h"
#ifdef __cplusplus
extern "C" {
#endif


enum Activation {
    Sigmoid=0,
    Tanh,
    Relu,
    Softmax,
    Linear
};


float fSigmoid(float val);
float fTanh(float val);
float fReLU(float val);
float fSoftmax(float val);
float fLinear(float val);


float fSigmoid_derivative(float val);
float fTanh_derivative(float val);
float fReLU_derivative(float val);
float fSoftmax_derivative(float val);
float fLinear_derivative(float val);


#ifdef __cplusplus
}
#endif


#endif
