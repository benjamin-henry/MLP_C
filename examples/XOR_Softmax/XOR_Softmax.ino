#include <MLP.h>

unsigned int input_shape = 2;
unsigned int layers_cfg[][2] = {
  {3, Relu},
  {4, Relu},
  {2, Softmax},
};

// XOR dataset
float X[][2] = {
  {0.f, 0.f},
  {0.f, 1.f},
  {1.f, 0.f},
  {1.f, 1.f},
};

float y_true[][2] = {
  {1.f, 0.f},
  {0.f, 1.f},
  {0.f, 1.f},
  {1.f, 0.f},
};

float input_buff[2] = {0.f};
float output_buff[2] = {0.f};

float learning_rate = .001f;

MLP mlp = mlp_from_cfg(input_shape, sizeof(layers_cfg) / sizeof(layers_cfg[0]), layers_cfg);


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
    float loss = train_on_batch(mlp, batch_size, output_shape, (float*)X, (float*)y_true, Categorical_Crossentropy, learning_rate);
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
