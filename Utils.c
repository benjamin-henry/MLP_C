#include "Utils.h"


void extract1dfrom2d(float* src, float* dst, unsigned int row, unsigned int width) {
    float *ptrIn = (float*)src;
     float *ptrOut = (float*)dst;
    for (int i = 0; i < width; i++) {
        *ptrOut++ = *(src + row*width + i);
    }
}


void delete_2d(float *arr[] ,unsigned int rows) {
  unsigned int i,j;
  for(i=0;i<rows;i++){
    free(arr[i]);
  }
  free(arr);
}
