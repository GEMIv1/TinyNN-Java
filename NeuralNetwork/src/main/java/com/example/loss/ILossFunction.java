package com.example.loss;

public interface ILossFunction {


    // Input: predictions Model predictions [batch_size x output_size], targets True labels [batch_size x output_size]
    // Output: Loss value (scalar)
    public double computeLoss(double[][] predictions, double[][] targets);

    // Input: predictions Model predictions [batch_size x output_size], targets True labels [batch_size x output_size]
    // Output: Compute the gradient of the loss with respect to predictions
    public double[][] computeGradient(double[][] predictions, double[][] targets);


}
