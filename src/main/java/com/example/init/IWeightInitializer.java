package com.example.init;

public interface IWeightInitializer {

    // Input: number of input neurons from the prev layer, number of output neurons in the current layer
    public double[][] init(int noInputs, int noOutputs);
}
