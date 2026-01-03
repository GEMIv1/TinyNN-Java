package com.example.layers;

import com.example.activations.IActivation;
import com.example.init.IWeightInitializer;

public class DenseLayer implements ILayer{

    private int inputSize;
    private int outputSize;
    private double[][] weights;      // [inputSize x outputSize]
    private double[] biases;         // [outputSize]
    private IActivation activation;

    // For backpropagation
    private double[][] input;        // Store input for backward pass
    private double[][] z;            // Pre-activation values
    private double[][] output;  
    
    private double[][] weightGradients;
    private double[] biasGradients;


    public DenseLayer(int inputSize, int outputSize, IWeightInitializer initializer, IActivation activation) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.activation = activation;
        this.weights = initializer.init(inputSize, outputSize);
        this.biases = new double[outputSize];        
        this.weightGradients = new double[inputSize][outputSize];
        this.biasGradients = new double[outputSize];
    }

    @Override
    public double[][] forward(double[][] input) {
        this.input = input; // needed for backprop  
        int batch_size = input.length;

        // z = (w*x) + b
        z = new double[batch_size][outputSize];

        for(int i=0;i<batch_size;i++){
            for(int j=0;j<outputSize;j++){
                double sum = biases[j];
                for(int k=0;k<inputSize;k++){
                    sum += (input[i][k] * weights[k][j]);
                }
                z[i][j] = sum;
            }
        }

        if(activation != null){
            output = activation.forward(z);
            return output;
        }
        else{
            output = z;
            return z;
        }
    }

    @Override
    public double[][] backward(double[][] outputGradient) {
        int batchSize = outputGradient.length;

        double[][] dz;
        if (activation != null) {
            dz = activation.backword(outputGradient);
        } else {
            dz = outputGradient;
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightGradients[i][j] = 0;
            }
        }

        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                for (int b = 0; b < batchSize; b++) {
                    weightGradients[i][j] += input[b][i] * dz[b][j];
                }
                weightGradients[i][j] /= batchSize;
            }
        }

        for (int j = 0; j < outputSize; j++) {
            biasGradients[j] = 0;
            for (int b = 0; b < batchSize; b++) {
                biasGradients[j] += dz[b][j];
            }
            biasGradients[j] /= batchSize; 
        }

        double[][] inputGradient = new double[batchSize][inputSize];
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                double sum = 0;
                for (int k = 0; k < outputSize; k++) {
                    sum += dz[i][k] * weights[j][k];
                }
                inputGradient[i][j] = sum;
            }
        }
        
        return inputGradient;
    }

    @Override
    public void updateParameters(double learningRate) {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weights[i][j] -= learningRate * weightGradients[i][j];
            }
        }
        
        for (int j = 0; j < outputSize; j++) {
            biases[j] -= learningRate * biasGradients[j];
        }
    }

    @Override
    public int getInputSize() {
        return inputSize;
    }

    @Override
    public int getOutputSize() {
        return outputSize;
    }

    public double[][] getWeights() {
        return weights;
    }
    
    public double[] getBiases() {
        return biases;
    }
    
    public double[][] getWeightGradients() {
        return weightGradients;
    }
    
    public double[] getBiasGradients() {
        return biasGradients;
    }


}
