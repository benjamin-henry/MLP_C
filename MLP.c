#include "MLP.h"

MLP mlp_from_cfg(unsigned int input_shape, unsigned int n_layers, unsigned int layers_cfg[][2]) {
  MLP mlp = (MLP)malloc(sizeof(mlp_t));
  mlp->_input_shape = input_shape;
  mlp->_n_layers = n_layers;


  mlp->_layers = (Dense*)malloc(sizeof(Dense) * mlp->_n_layers);
  mlp->_layers[0] = build_dense(input_shape, layers_cfg[0][0], layers_cfg[0][1]);

  for (int i = 1; i < n_layers; i++) {
    mlp->_layers[i] = build_dense(mlp->_layers[i - 1]->_units, layers_cfg[i][0], layers_cfg[i][1]);
  }
  return mlp;
}

void randomize_mlp(MLP mlp) {
  for (int i = 0; i < mlp->_n_layers; i++) {
    randomize_weights(mlp->_layers[i], -1., 1.);
  }
}

void mlp_predict(MLP mlp, float * input_data, float * output) {
  unsigned int i;

  dense_forward(mlp->_layers[0], input_data);
  for (i = 1; i < mlp->_n_layers; i++) {
    dense_forward(mlp->_layers[i], mlp->_layers[i - 1]->_output);
  }

  for (i = 0; i < mlp->_layers[mlp->_n_layers - 1]->_units; i++) {
    output[i] = (float)mlp->_layers[mlp->_n_layers - 1]->_output[i];
  }
}


float train_on_batch(MLP mlp, unsigned int batch_size, unsigned int output_shape, float* X, float* y_true, unsigned int loss, float learning_rate) {
  float * mlp_input_buf = (float*)calloc(mlp->_input_shape, sizeof(float));
  float * y_true_buf = (float*)calloc(output_shape, sizeof(float));
  float * y_pred_buf = (float*)calloc(output_shape, sizeof(float));
  float * error_buf = (float*)calloc(output_shape, sizeof(float));

  float ** derivatives = (float**)calloc(mlp->_n_layers, sizeof(float*));
  float ** local_gradient = (float**)calloc(mlp->_n_layers, sizeof(float*));
  float ** dB = (float**)calloc(mlp->_n_layers, sizeof(float*));
  float *** dW = (float***)calloc(mlp->_n_layers, sizeof(float**));

  unsigned int i_sample, i_layer, i_unit;
  unsigned int i, j, k;

  float _loss = 0.;

  // allocate memory for stuff
  for (i_layer = 0; i_layer < mlp->_n_layers; i_layer++) {
    derivatives[i_layer] = (float*)calloc(mlp->_layers[i_layer]->_units, sizeof(float));
    local_gradient[i_layer] = (float*)calloc(mlp->_layers[i_layer]->_units, sizeof(float));
    dB[i_layer] = (float*)calloc(mlp->_layers[i_layer]->_units, sizeof(float));

    dW[i_layer] = (float**)calloc(mlp->_layers[i_layer]->_input_shape, sizeof(float*));
    for (j = 0; j < mlp->_layers[i_layer]->_input_shape; j++) {
      dW[i_layer][j] = (float*)calloc(mlp->_layers[i_layer]->_units, sizeof(float));
    }
  }

  // for each pair in X-y_true pairs
  for (i_sample = 0; i_sample < batch_size; i_sample++) {
    // get the i_sample-th of X and y_true
    for (i = 0; i < mlp->_input_shape; i++) {
      mlp_input_buf[i] = *(X + i_sample * mlp->_input_shape + i);
    }
    for (i = 0; i < output_shape; i++) {
      y_true_buf[i] = *(y_true + i_sample * output_shape + i);
    }

    // forward propagation
    mlp_predict(mlp, mlp_input_buf, y_pred_buf);

    // compute error
    for (i = 0; i < output_shape; i++) {
      error_buf[i] = y_true_buf[i] - y_pred_buf[i];
      switch (loss) {
        case MSE:
          _loss += error_buf[i] * error_buf[i];
          break;

        case Categorical_Crossentropy:
          _loss -= y_true_buf[i] * (float)log(y_pred_buf[i]);
          break;
          
        case Binary_Crossentropy:
          _loss -= y_true_buf[i] * (float)log(y_pred_buf[i]) + (1.f - y_true_buf[i]) * (float)log(1 - y_pred_buf[i]);
          break;

        default:
          break;
      }
    }

    // compute derivative for each layer
    for (i_layer = 0; i_layer < mlp->_n_layers; i_layer++) {
      dense_activation_derivative(mlp->_layers[i_layer], derivatives[i_layer]);
    }

    // compute last layer delta
    for (i_unit = 0; i_unit < mlp->_layers[mlp->_n_layers - 1]->_units; i_unit++) {
      local_gradient[mlp->_n_layers - 1][i_unit] = 2. * error_buf[i_unit] * derivatives[mlp->_n_layers - 1][i_unit];
    }

    float err = 0.;
    // //compute remaining layers delta
    for (i_layer = mlp->_n_layers - 1; i_layer > 0; i_layer--) {
      for (i_unit = 0; i_unit < mlp->_layers[i_layer - 1]->_units; i_unit++) {
        err = 0.;
        for (k = 0; k < mlp->_layers[i_layer]->_units; k++) {
          err += local_gradient[i_layer][k] * mlp->_layers[i_layer]->_w[i_unit][k];
          //Serial.println(err,12);
        }
        local_gradient[i_layer - 1][i_unit] = err * derivatives[i_layer - 1][i_unit];
        //Serial.println(err,12);
      }
    }

    for (i_layer = mlp->_n_layers - 1; i_layer > 0; i_layer--) {
      for (i_unit = 0; i_unit < mlp->_layers[i_layer]->_units; i_unit++) {
        for (k = 0 ; k < mlp->_layers[i_layer]->_input_shape; k++) {
          dW[i_layer][k][i_unit] += mlp->_layers[i_layer - 1]->_output[k] * local_gradient[i_layer][i_unit];
        }
        dB[i_layer][i_unit] += local_gradient[i_layer][i_unit];
      }
      //Serial.println(dB[i_layer][i_unit]);
    }
    for (i_unit = 0; i_unit < mlp->_layers[0]->_units; i_unit++) {
      for (k = 0 ; k < mlp->_layers[0]->_input_shape; k++) {
        dW[0][k][i_unit] += mlp_input_buf[k] * local_gradient[0][i_unit];
      }
      dB[0][i_unit] += local_gradient[0][i_unit];
    }

    // set local gradients to zero for next iter
    for (i_layer = 0; i_layer < mlp->_n_layers; i_layer++) {
      for (j = 0; j < mlp->_layers[i_layer]->_units; j++) {
        local_gradient[i_layer][j] = 0.0f;
      }
    }
  }

  for (i_layer = 0; i_layer < mlp->_n_layers; i_layer++) {
    for (i_unit = 0; i_unit < mlp->_layers[i_layer]->_units; i_unit++) {
      for (k = 0 ; k < mlp->_layers[i_layer]->_input_shape; k++) {
        mlp->_layers[i_layer]->_w[k][i_unit] += learning_rate * dW[i_layer][k][i_unit];
      }
      mlp->_layers[i_layer]->_b[i_unit] += learning_rate * dB[i_layer][i_unit];
    }
  }

  for (i_layer = 0; i_layer < mlp->_n_layers; i_layer++) {
    free(derivatives[i_layer]);
    free(local_gradient[i_layer]);
    free(dB[i_layer]);
    for (j = 0; j < mlp->_layers[i_layer]->_input_shape; j++) {
      free(dW[i_layer][j]);
    }
    free(dW[i_layer]);
  }
  free(dW);
  free(dB);
  free(derivatives);
  free(local_gradient);

  free(mlp_input_buf);
  free(y_true_buf);
  free(y_pred_buf);
  free(error_buf);
  return _loss / (batch_size*output_shape);
}
