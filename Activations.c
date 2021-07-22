#include "Activations.h"
#include "Dense.h"


float fSigmoid(float val) {
    return 1. / (1.+exp(-val));
}


float fTanh(float val) {
    return tanh(val);
}

float fReLU(float val) {
    return val > 0.f ? val : 0.f;
}


float fSoftmax(float val) {
    ;
}


float fLinear(float val) {
  return val;
}


float fSigmoid_derivative(float val) {
    float tmp = fSigmoid(val);
    return tmp * (1.f - tmp);
}


float fTanh_derivative(float val) {
    float tanh_sq = fTanh(val);
    tanh_sq *= tanh_sq;
    return 1.f - tanh_sq;
}

float fReLU_derivative(float val) {
    return val > 0. ? 1.f : 0.f;
}


float fSoftmax_derivative(float val) {
    ;
}

float fLinear_derivative(float val) {
  return 1;
}
