package com.example.activations;

public class Tanh implements IActivation{

    private double[][] input;


    @Override
    public double[][] forward(double[][] z) {
        this.input = z; // store for backpropagation
        int batches = z.length;
        int neurons = z[0].length;
        double[][] output = new double[batches][neurons];

        for(int i=0;i<batches;i++){
            for(int j=0;j<neurons;j++){
                output[i][j] = Math.tanh(z[i][j]);
            }
        }
        return output;
    }

    @Override
    public double[][] backword(double[][] da) {
        int batches = da.length;
        int neurons = da[0].length;
        double[][] inputGrad = new double[batches][neurons];

        for(int i=0;i<batches;i++){
            for(int j=0;j<neurons;j++){

                // Derivative of tanh: 1 - tanh^2(x)
                double tanhValue = input[i][j];
                double drev = 1 - (tanhValue * tanhValue);
                inputGrad[i][j] = da[i][j]*drev;
            }
        }
        return inputGrad;
    }

}
