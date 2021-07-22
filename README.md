# MLP_C
Multi Layer Perceptron implementation in C ( for Arduino or other MCUs )


Available Activations
```c
enum Activation {
    Sigmoid=0,
    Tanh,
    Relu,
    Softmax,
    Linear
};
```

Available Losses 
```c
enum Loss {
  Categorical_Crossentropy = 0,
  Binary_Crossentropy,
  MSE,
};
```

To see examples, please go to the examples folder.
