package com.example.activations;

public class Sigmoid implements IActivation{

    private double[][] output;

    @Override
    public double[][] forward(double[][] z) {
        int batches = z.length;
        int neurons = z[0].length;
        output = new double[batches][neurons];

        for (int i = 0; i < batches; i++) {
            for (int j = 0; j < neurons; j++) {
                output[i][j] = 1.0 / (1.0 + Math.exp(-z[i][j]));
            }
        }
        return output;
    }


    @Override
    public double[][] backword(double[][] da) {
        int batches = da.length;
        int neurons = da[0].length;
        double[][] inputGrad = new double[batches][neurons];

        for (int i = 0; i < batches; i++) {
            for (int j = 0; j < neurons; j++) {
                
                double sigmoidValue = output[i][j];
                double derivative = sigmoidValue * (1.0 - sigmoidValue);
                inputGrad[i][j] = da[i][j] * derivative;
            }
        }
        return inputGrad;
    }

}