#include <MLP.h>

unsigned int input_shape = 2;
unsigned int layers_cfg[][2] = {
  {3, Relu},
  {4, Relu},
  {1, Sigmoid},
};

// XOR dataset
float X[4][2] = {
  {0., 0.},
  {0., 1.},
  {1., 0.},
  {1., 1.},
};

float y_true[4][1] = {
  {0.},
  {1.},
  {1.},
  {0.},
};

float input_buff[2] = {0.};
float output_buff[1] = {0.};

MLP mlp = mlp_from_cfg(input_shape, sizeof(layers_cfg) / sizeof(layers_cfg[0]), layers_cfg);

float learning_rate = .1;

void setup() {
  Serial.begin(115200);
  randomize_mlp(mlp);

  unsigned int batch_size = sizeof(X) / sizeof(X[0]);
  unsigned int output_shape = mlp->_layers[mlp->_n_layers - 1]->_units;
  unsigned int cnt = 99;

  Serial.print("batch");
  Serial.print('\t');
  Serial.println("loss");
  for (unsigned int i = 0; i < 50000; i++) {
    float loss = train_on_batch(mlp, batch_size, output_shape, (float*)X, (float*)y_true, Binary_Crossentropy, learning_rate);
    if (!cnt) {
      cnt = 99;
      Serial.print(i + 1);
      Serial.print('\t');
      Serial.println(loss, 12);
    } else {
      cnt--;
    }
  }

  Serial.println();

  Serial.print("y_true");
  Serial.print('\t');
  Serial.println("y_pred");
  for (unsigned int i_sample = 0; i_sample < batch_size; i_sample++) {
    extract1dfrom2d((float*)X, (float*)input_buff, i_sample , mlp->_input_shape);
    mlp_predict(mlp, input_buff, output_buff);
    for (int j = 0; j < output_shape; j++) {
      Serial.print(y_true[i_sample][j], 5);
      Serial.print('\t');
      Serial.println(output_buff[j], 5);
    }
    Serial.println();
  }
}

void loop() {
  // put your main code here, to run repeatedly:

}
