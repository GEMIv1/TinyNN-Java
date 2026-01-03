package com.example.init;

import java.util.Random;

public class Xavier implements IWeightInitializer {

    private final Random random = new Random();

    @Override
    public double[][] init(int noInputs, int noOutputs) {
        
        double stdDev = Math.sqrt(2.0 / (noInputs + noOutputs));
        double[][] weights = new double[noInputs][noOutputs];

        for (int i = 0; i < noInputs; i++) {
            for (int j = 0; j < noOutputs; j++) {
                weights[i][j] = random.nextGaussian() * stdDev;
            }
        }
        return weights;
    }

}