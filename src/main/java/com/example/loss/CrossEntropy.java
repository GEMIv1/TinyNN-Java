package com.example.loss;

public class CrossEntropy implements ILossFunction {

    private static final double EPSILON = 1e-15;
    
    @Override
    public double computeLoss(double[][] predictions, double[][] targets) {

        if (predictions == null || targets == null) {
            throw new IllegalArgumentException("Predictions and targets cannot be null");
        }
        
        if (predictions.length == 0 || targets.length == 0) {
            throw new IllegalArgumentException("Predictions and targets cannot be empty");
        }
        
        if (predictions.length != targets.length || predictions[0].length != targets[0].length) {
            throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
        }


        int batchSize = predictions.length;
        int numClasses = predictions[0].length;
        
        double totalLoss = 0.0;

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < numClasses; j++) {
                double pred = Math.max(EPSILON, Math.min(1.0 - EPSILON, predictions[i][j]));
                double target = targets[i][j];
                
                
                totalLoss += -target * Math.log(pred) - (1.0 - target) * Math.log(1.0 - pred);
            }
        }

        return totalLoss / (batchSize * numClasses);
    }

    @Override
    public double[][] computeGradient(double[][] predictions, double[][] targets) {

    if (predictions == null || targets == null) {
        throw new IllegalArgumentException("Predictions and targets cannot be null");
    }
    
    if (predictions.length == 0 || targets.length == 0) {
        throw new IllegalArgumentException("Predictions and targets cannot be empty");
    }
    
    if (predictions.length != targets.length || predictions[0].length != targets[0].length) {
        throw new IllegalArgumentException("Predictions and targets must have the same dimensions");
    }

    int batchSize = predictions.length;
    int numClasses = predictions[0].length;
    
    double[][] gradient = new double[batchSize][numClasses];

    // For binary cross-entropy with sigmoid activation,
    // the gradient simplifies to: prediction - target
    for (int i = 0; i < batchSize; i++) {
        for (int j = 0; j < numClasses; j++) {
            gradient[i][j] = (predictions[i][j] - targets[i][j]) / batchSize;
        }
    }

    return gradient;
}

}