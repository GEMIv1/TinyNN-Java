package com.example.init;

import java.util.Random;

public class He implements IWeightInitializer{
    private Random random;
    
    public He() {
        this.random = new Random();
    }
    
    public He(long seed) {
        this.random = new Random(seed);
    }

    @Override
    public double[][] init(int noInputs, int noOutputs) {
        if (noInputs <= 0 || noOutputs <= 0) {
            throw new IllegalArgumentException("Number of inputs and outputs must be positive");
        }

        double[][] weights = new double[noInputs][noOutputs];
        
        double stddev = Math.sqrt(2.0 / noInputs);

        for (int i = 0; i < noInputs; i++) {
            for (int j = 0; j < noOutputs; j++) {
                weights[i][j] = random.nextGaussian() * stddev;
            }
        }

        return weights;
    }
}
