package com.example.utils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class DataSplitter {
    private Random random;

    public DataSplitter() {
        this.random = new Random();
    }

    public DataSplitter(long seed){
        this.random = new Random(seed);
    }

    public double[][][] trainTestSplit(double[][] X, double[][] y, double testRatio){
        if(testRatio <= 0 || testRatio >=1){
            throw new IllegalArgumentException("Train ratio must be between 0 and 1");
        }

        int totalSamples = X.length;
        int trainSize = (int) (totalSamples * (1-testRatio));

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);

        double[][] X_train = new double[trainSize][];
        double[][] X_test = new double[totalSamples - trainSize][];
        double[][] y_train = new double[trainSize][];
        double[][] y_test = new double[totalSamples - trainSize][];

        for(int i=0;i<trainSize;i++){
            int idx = indices.get(i);
            X_train[i] = X[idx];
            y_train[i] = y[idx];
        }

        for (int i=trainSize;i<totalSamples;i++) {
            int idx = indices.get(i);
            X_test[i-trainSize] = X[idx];
            y_test[i-trainSize] = y[idx];
        }
        
        return new double[][][] {X_train, X_test, y_train, y_test};
    }

    
    public double[][][][] createBatches(double[][] X, double[][] y, int batchSize) {
        int totalSamples = X.length;
        int numBatches = (int) Math.ceil((double) totalSamples / batchSize);
        
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < totalSamples; i++) {
            indices.add(i);
        }

        double[][][][] batches = new double[numBatches][][][];
        
        for (int b = 0; b < numBatches; b++) {
            int start = b * batchSize;
            int end = Math.min(start + batchSize, totalSamples);
            int currentBatchSize = end - start;
            
            double[][] X_batch = new double[currentBatchSize][];
            double[][] y_batch = new double[currentBatchSize][];
            
            for (int i = 0; i < currentBatchSize; i++) {
                int idx = indices.get(start + i);
                X_batch[i] = X[idx];
                y_batch[i] = y[idx];
            }
            
            batches[b] = new double[][][] {X_batch, y_batch};
        }
        
        return batches;
    }    
}
