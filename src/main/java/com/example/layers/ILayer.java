package com.example.layers;

public interface ILayer {
    

    // Input: Input data [batch_size x input_features]
    // Output: data [batch_size x output_features]
    public double[][] forward(double[][] input);

    // Output: outputGradient Gradient from the next layer [batch_size x output_features]
    // Input: Gradient to pass to previous layer [batch_size x input_features]
    public double[][] backward(double[][] outputGradient);

    // Update weights using gradients, lr
    public void updateParameters(double learningRate);

    public int getInputSize();

    public int getOutputSize();

}
