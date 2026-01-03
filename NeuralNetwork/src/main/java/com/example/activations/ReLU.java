package com.example.activations;

public class ReLU implements IActivation{

    private double[][] input;

    @Override
    public double[][] forward(double[][] z) {
        this.input = z; // store for backpropagation
        int batches = z.length;
        int neurons = z[0].length;
        double[][] output = new double[batches][neurons];

        for(int i=0;i<batches;i++){
            for(int j=0;j<neurons;j++){
                output[i][j] = Math.max(0, z[i][j]);
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
                double drev = (input[i][j] > 0)?1:0;
                inputGrad[i][j] = da[i][j]*drev;
            }
        }
        return inputGrad;
    }

}
