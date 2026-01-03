package com.example.loss;

public class MeanAbsoluteError implements ILossFunction {

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
        int outputSize = predictions[0].length;
        
        double totalLoss = 0.0;
        
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                totalLoss += Math.abs(predictions[i][j] - targets[i][j]);
            }
        }
        
        return totalLoss / (batchSize * outputSize);
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
        int outputSize = predictions[0].length;
        
        double[][] gradient = new double[batchSize][outputSize];
        
        // Gradient of MAE: sign(prediction - target) / (batch_size * output_size)
        double normalizationFactor = 1.0 / (batchSize * outputSize);
        
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                double diff = predictions[i][j] - targets[i][j];
                
                // Compute sign of the difference
                if (diff > 0) {
                    gradient[i][j] = normalizationFactor;
                } else if (diff < 0) {
                    gradient[i][j] = -normalizationFactor;
                } else {
                    // When diff == 0, gradient is technically undefined
                    // Common practice is to set it to 0
                    gradient[i][j] = 0.0;
                }
            }
        }
        
        return gradient;
    }
}