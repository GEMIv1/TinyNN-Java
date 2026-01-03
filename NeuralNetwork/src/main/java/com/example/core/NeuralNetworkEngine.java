package com.example.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.example.layers.ILayer;
import com.example.loss.ILossFunction;

public class NeuralNetworkEngine {

    private List<ILayer> layers;
    private Random random;

    public NeuralNetworkEngine() {
        this.layers = new ArrayList<>();
        this.random = new Random(42);
    }

    public void addLayer(ILayer layer) {
        if (layer == null) {
            throw new IllegalArgumentException("Layer cannot be null");
        }
        
        if (!layers.isEmpty()) {
            ILayer lastLayer = layers.get(layers.size() - 1);
            if (lastLayer.getOutputSize() != layer.getInputSize()) {
                throw new IllegalArgumentException(
                    "Layer input size (" + layer.getInputSize() + 
                    ") must match previous layer output size (" + lastLayer.getOutputSize() + ")"
                );
            }
        }
        
        layers.add(layer);
    }

    public double[][] forward(double[][] input) {
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network has no layers");
        }
        if (input == null || input.length == 0) {
            throw new IllegalArgumentException("Input cannot be null or empty");
        }

        double[][] output = input;
        for (ILayer layer : layers) {
            output = layer.forward(output);
        }
        return output;
    }

    public double[][] backward(double[][] lossGradient) {
        if (lossGradient == null || lossGradient.length == 0) {
            throw new IllegalArgumentException("Loss gradient cannot be null or empty");
        }

        double[][] gradient = lossGradient;
        
        // Backpropagate through layers in reverse order
        for (int i = layers.size() - 1; i >= 0; i--) {
            gradient = layers.get(i).backward(gradient);
        }
        
        return gradient;
    }

    public void updateParameters(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        
        for (ILayer layer : layers) {
            layer.updateParameters(learningRate);
        }
    }

    public void resetWeights() {
        for (ILayer layer : layers) {
            try {
                java.lang.reflect.Method method = layer.getClass().getMethod("resetWeights");
                method.invoke(layer);
            } catch (Exception e) {
                System.err.println("Warning: Could not reset weights for layer " + 
                                   layer.getClass().getSimpleName());
            }
        }
    }

    public double[][] predict(double[][] inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("Inputs cannot be null or empty");
        }
        if (layers.isEmpty()) {
            throw new IllegalStateException("Network has no layers");
        }

        return forward(inputs);
    }

    public double evaluate(double[][] inputs, double[][] targets, ILossFunction lossFunction) {
        if (lossFunction == null) {
            throw new IllegalArgumentException("Loss function cannot be null");
        }
        if (inputs == null || targets == null) {
            throw new IllegalArgumentException("Inputs and targets cannot be null");
        }
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of input samples must match number of target samples");
        }

        double[][] predictions = predict(inputs);
        return lossFunction.computeLoss(predictions, targets);
    }

    public double computeAccuracy(double[][] inputs, double[][] targets) {
        double[][] predictions = predict(inputs);
        int correct = 0;
        
        for (int i = 0; i < predictions.length; i++) {
            int predictedClass = argmax(predictions[i]);
            int trueClass = argmax(targets[i]);
            if (predictedClass == trueClass) {
                correct++;
            }
        }
        
        return (double) correct / predictions.length * 100.0;
    }

    private int argmax(double[] array) {
        int maxIdx = 0;
        double maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    public int getLayerCount() {
        return layers.size();
    }

    public ILayer getLayer(int index) {
        if (index < 0 || index >= layers.size()) {
            throw new IndexOutOfBoundsException("Layer index out of bounds");
        }
        return layers.get(index);
    }

    public List<ILayer> getLayers() {
        return new ArrayList<>(layers);
    }

    public void clearLayers() {
        layers.clear();
    }

    public Random getRandom() {
        return random;
    }

    public void setSeed(long seed) {
        this.random = new Random(seed);
    }

    public void printSummary() {
        System.out.println("Neural Network Architecture:");
        System.out.println("========================================");
        System.out.println("Total Layers: " + layers.size());
        System.out.println("----------------------------------------");
        
        int totalParams = 0;
        for (int i = 0; i < layers.size(); i++) {
            ILayer layer = layers.get(i);
            int layerParams = layer.getInputSize() * layer.getOutputSize() + layer.getOutputSize();
            totalParams += layerParams;
            
            System.out.printf("Layer %d: [%d -> %d] (%s) - Parameters: %d%n", 
                i + 1, 
                layer.getInputSize(), 
                layer.getOutputSize(),
                layer.getClass().getSimpleName(),
                layerParams
            );
        }
        System.out.println("----------------------------------------");
        System.out.println("Total Parameters: " + totalParams);
        System.out.println("========================================");
    }
}