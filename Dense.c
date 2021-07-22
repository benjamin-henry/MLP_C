#include "Dense.h"


Dense build_dense(unsigned int input_shape, unsigned int units, unsigned int activation) {
  Dense layer = (Dense)malloc(sizeof(dense_t));
  layer->_units = units;
  layer->_input_shape = input_shape;

  layer->_activation = activation;

  // allocate memory for weights
  layer->_w = (float**)calloc(input_shape,sizeof(float*));
  for (int i = 0; i < input_shape; i++) {
    layer->_w[i] = (float*)calloc(units,sizeof(float));
  }

  // allocate memory for biases
  layer->_b = (float*)calloc(units,sizeof(float));

  // allocate memory for "pre-output"
  layer->_z = (float*)calloc(units,sizeof(float));

  // allocate memory for activated output
  layer->_output = (float*)calloc(units,sizeof(float));


  return layer;
}


void randomize_weights(Dense layer, float minimum, float maximum) {
  for (unsigned int row = 0; row < layer->_input_shape; row++) {
    for (unsigned int col = 0; col < layer->_units; col++) {
      layer->_w[row][col] = (((float)rand() / (float)(RAND_MAX)) * (maximum - minimum) + minimum) / layer->_units;
    }
  }
}


void dense_fSigmoid(Dense layer) {
    for(int i = 0; i < layer->_units; i++) {
        layer->_output[i] = fSigmoid(layer->_output[i]);
    }
}

void dense_fLinear(Dense layer) {
  for(int i = 0; i < layer->_units; i++) {
        layer->_output[i] = layer->_output[i];
    }
}

void dense_fTanh(Dense layer) {
    for(int i = 0; i < layer->_units; i++) {
        layer->_output[i] = fTanh(layer->_output[i]);
    }
}

void dense_fReLU(Dense layer) {
    for(int i = 0; i < layer->_units; i++) {
        layer->_output[i] = fReLU(layer->_output[i]);
    }
}

void dense_fSoftmax(Dense layer) {
    float accu = 0.;
    unsigned int i;
    for(i = 0; i < layer->_units; i++) {
        layer->_output[i] = exp(layer->_output[i]);
        accu += layer->_output[i];
    }
    
    for(i = 0; i < layer->_units; i++) {
        layer->_output[i] /= accu;
    }
}

void dense_forward(Dense layer, float * input_data) {
  float accu = 0.;
  int i, j;
  for (i = 0; i < layer->_units; i++) {
    accu = 0.;
    for (j = 0; j < layer->_input_shape; j++) {
      accu += input_data[j] * layer->_w[j][i];
    }
    layer->_z[i] = accu+layer->_b[i];
  }
  
  for (i = 0; i < layer->_units; i++) {
    layer->_output[i] = layer->_z[i];
  }
  dense_activation(layer);
  
}

void dense_activation(Dense layer) {
  switch (layer->_activation) {
    case Linear:
      dense_fLinear(layer);
      break;

    case Sigmoid:
      dense_fSigmoid(layer);
      break;

    case Tanh:
      dense_fTanh(layer);
      break;

    case Relu:
      dense_fReLU(layer);
      break;

    case Softmax:
      dense_fSoftmax(layer);
      break;

    default:
      break;
  }
}

void dense_fSigmoid_derivative(Dense layer, float * output) {
  for(int i = 0; i< layer->_units; i++) {
    output[i] = fSigmoid_derivative(layer->_z[i]);
  }
}

void dense_fLinear_derivative(Dense layer, float * output) {
  for(int i = 0; i< layer->_units; i++) {
    output[i] = fLinear_derivative(layer->_z[i]);
  }
}

void dense_fTanh_derivative(Dense layer, float * output) {
  for(int i = 0; i< layer->_units; i++) {
    output[i] = fTanh_derivative(layer->_z[i]);
  }
}

void dense_fReLU_derivative(Dense layer, float * output) {
  for(int i = 0; i< layer->_units; i++) {
    output[i] = fReLU_derivative(layer->_z[i]);
  }
}

void dense_fSoftmax_derivative(Dense layer, float * output) {
  for(int i = 0; i< layer->_units; i++) {
    output[i] = layer->_output[i] * (1.f - layer->_output[i]);
  }
}

void dense_activation_derivative(Dense layer, float * output) {
  switch (layer->_activation) {
    case Linear:
      dense_fLinear_derivative(layer,output);
      break;

    case Sigmoid:
      dense_fSigmoid_derivative(layer,output);
      break;

    case Tanh:
      dense_fTanh_derivative(layer,output);
      break;

    case Relu:
      dense_fReLU_derivative(layer,output);
      break;

    case Softmax:
      dense_fSoftmax_derivative(layer,output);
      break;

    default:
      break;
  }
}
