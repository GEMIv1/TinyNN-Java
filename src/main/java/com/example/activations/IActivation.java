package com.example.activations;

public interface IActivation {
    
    // double because of the batches row ---> batch size, col ---> number of neurons
    
    // Input: z = (w * x) + b -> (pre-activation value) [Weighted sum]
    // Output: transformation
    double[][] forward(double[][] z);

    // Input: da/de output gradient
    // Output: gradient with respect to the weighted sum
    double[][] backword(double[][] da);

}
