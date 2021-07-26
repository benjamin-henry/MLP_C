#ifndef MLP_H
#define MLP_H

#include "math.h"
#include "Arduino.h"

#include "Activations.h"
#include "Dense.h"
#include "Utils.h"
#include "Losses.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mlp_t {
    unsigned int _input_shape;
    unsigned int _n_layers;
    Dense* _layers;
} mlp_t;

typedef struct mlp_t* MLP;

MLP mlp_from_cfg(unsigned int input_shape, unsigned int n_layers, unsigned int layers_cfg[][2]);
void randomize_mlp(MLP mlp);


void mlp_predict(MLP mlp, float * input_data, float * output);

// returns loss
float train_on_batch(MLP mlp, unsigned int batch_size, unsigned int output_shape, float* X, float* y_true, unsigned int loss, float learning_rate);



#ifdef __cplusplus
}
#endif

unsigned int mlp_save_config(MLP mlp, unsigned int address);
unsigned int mlp_save_weights(MLP mlp, unsigned int address);
unsigned int mlp_load_weights(MLP mlp, unsigned int address);

#endif
