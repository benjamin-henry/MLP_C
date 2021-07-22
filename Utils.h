#ifndef MLP_UTILS_H
#define MLP_UTILS_H
#include "Arduino.h"

#ifdef __cplusplus
extern "C" {
#endif

void extract1dfrom2d(float* src, float* dst, unsigned int index, unsigned int shape);


#ifdef __cplusplus
}
#endif

#endif
