package com.example.core;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.example.loss.ILossFunction;

public class NetworkTrainer {

    private NeuralNetworkEngine engine;
    private ILossFunction lossFunction;
    private double learningRate;
    private int epochs;
    private int batchSize;
    private boolean verbose;
    private Random random;
    
    private List<Double> trainingLossHistory;
    private List<Double> validationLossHistory;

    public NetworkTrainer(NeuralNetworkEngine engine) {
        if (engine == null) {
            throw new IllegalArgumentException("Neural network engine cannot be null");
        }
        this.engine = engine;
        this.learningRate = 0.01;
        this.epochs = 100;
        this.batchSize = 32;
        this.verbose = true;
        this.random = new Random();
        this.trainingLossHistory = new ArrayList<>();
        this.validationLossHistory = new ArrayList<>();
    }

    public NetworkTrainer(NeuralNetworkEngine engine, long seed) {
        this(engine);
        this.random = new Random(seed);
    }

    public void setLossFunction(ILossFunction lossFunction) {
        if (lossFunction == null) {
            throw new IllegalArgumentException("Loss function cannot be null");
        }
        this.lossFunction = lossFunction;
    }

    public void setLearningRate(double learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("Learning rate must be positive");
        }
        this.learningRate = learningRate;
    }

    public void setEpochs(int epochs) {
        if (epochs <= 0) {
            throw new IllegalArgumentException("Number of epochs must be positive");
        }
        this.epochs = epochs;
    }

    public void setBatchSize(int batchSize) {
        if (batchSize <= 0) {
            throw new IllegalArgumentException("Batch size must be positive");
        }
        this.batchSize = batchSize;
    }

    public void setVerbose(boolean verbose) {
        this.verbose = verbose;
    }

    private double[][][] shuffleData(double[][] inputs, double[][] targets) {
        int numSamples = inputs.length;
        int[] indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            indices[i] = i;
        }
        
        for (int i = numSamples - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        double[][] shuffledInputs = new double[numSamples][];
        double[][] shuffledTargets = new double[numSamples][];
        for (int i = 0; i < numSamples; i++) {
            shuffledInputs[i] = inputs[indices[i]];
            shuffledTargets[i] = targets[indices[i]];
        }
        
        return new double[][][] { shuffledInputs, shuffledTargets };
    }

    public void train(double[][] inputs, double[][] targets) {
        train(inputs, targets, null, null);
    }

    public void train(double[][] inputs, double[][] targets, double[][] valInputs, double[][] valTargets) {
        if (lossFunction == null) {
            throw new IllegalStateException("Loss function must be set before training");
        }
        if (inputs == null || targets == null) {
            throw new IllegalArgumentException("Inputs and targets cannot be null");
        }
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of input samples must match number of target samples");
        }
        if (engine.getLayerCount() == 0) {
            throw new IllegalStateException("Network must have at least one layer");
        }

        trainingLossHistory.clear();
        validationLossHistory.clear();

        int numSamples = inputs.length;
        int numBatches = (int) Math.ceil((double) numSamples / batchSize);
        
        boolean hasValidation = (valInputs != null && valTargets != null);

        if (verbose) {
            System.out.println("Starting training...");
            System.out.println("Samples: " + numSamples + ", Batch Size: " + batchSize + ", Epochs: " + epochs);
            System.out.println("Learning Rate: " + learningRate);
            if (hasValidation) {
                System.out.println("Validation Samples: " + valInputs.length);
            }
            System.out.println("========================================");
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            double[][][] shuffled = shuffleData(inputs, targets);
            double[][] shuffledInputs = shuffled[0];
            double[][] shuffledTargets = shuffled[1];
            
            double totalLoss = 0.0;

            for (int batch = 0; batch < numBatches; batch++) {
                int startIdx = batch * batchSize;
                int endIdx = Math.min(startIdx + batchSize, numSamples);
                int currentBatchSize = endIdx - startIdx;

                double[][] batchInputs = new double[currentBatchSize][];
                double[][] batchTargets = new double[currentBatchSize][];
                
                for (int i = 0; i < currentBatchSize; i++) {
                    batchInputs[i] = shuffledInputs[startIdx + i];
                    batchTargets[i] = shuffledTargets[startIdx + i];
                }

                double[][] predictions = engine.forward(batchInputs);

                double batchLoss = lossFunction.computeLoss(predictions, batchTargets);
                totalLoss += batchLoss * currentBatchSize;

                double[][] lossGradient = lossFunction.computeGradient(predictions, batchTargets);

                engine.backward(lossGradient);
                
                engine.updateParameters(learningRate);
            }

            double avgLoss = totalLoss / numSamples;
            trainingLossHistory.add(avgLoss);

            if (verbose && ((epoch + 1) % 10 == 0 || epoch == 0 || epoch == epochs - 1)) {
                StringBuilder output = new StringBuilder();
                output.append(String.format("Epoch %d/%d - Loss: %.6f", epoch + 1, epochs, avgLoss));
                
                if (hasValidation) {
                    double valLoss = engine.evaluate(valInputs, valTargets, lossFunction);
                    validationLossHistory.add(valLoss);
                    output.append(String.format(" - Val Loss: %.6f", valLoss));
                }
                
                System.out.println(output.toString());
            } else if (hasValidation) {
                double valLoss = engine.evaluate(valInputs, valTargets, lossFunction);
                validationLossHistory.add(valLoss);
            }
        }

        if (verbose) {
            System.out.println("========================================");
            System.out.println("Training completed!");
            System.out.println("Final Training Loss: " + trainingLossHistory.get(trainingLossHistory.size() - 1));
            if (hasValidation && !validationLossHistory.isEmpty()) {
                System.out.println("Final Validation Loss: " + validationLossHistory.get(validationLossHistory.size() - 1));
            }
        }
    }

    public List<Double> getTrainingLossHistory() {
        return new ArrayList<>(trainingLossHistory);
    }

    public List<Double> getValidationLossHistory() {
        return new ArrayList<>(validationLossHistory);
    }

    public double getLearningRate() {
        return learningRate;
    }

    public int getEpochs() {
        return epochs;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public ILossFunction getLossFunction() {
        return lossFunction;
    }
}