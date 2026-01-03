package com.example.init;

import java.util.Random;

public class RandomUniform implements IWeightInitializer {

    private double minValue;
    private double maxValue;
    private Random random;

    public RandomUniform(double minValue, double maxValue, long seed) {
        if (minValue >= maxValue) {
            throw new IllegalArgumentException("minValue must be less than maxValue");
        }
        this.minValue = minValue;
        this.maxValue = maxValue;
        this.random = new Random(seed);
    }

    @Override
    public double[][] init(int noInputs, int noOutputs) {
        if (noInputs <= 0 || noOutputs <= 0) {
            throw new IllegalArgumentException("Number of inputs and outputs must be positive");
        }

        double[][] weights = new double[noInputs][noOutputs];
        double range = maxValue - minValue;

        for (int i = 0; i < noInputs; i++) {
            for (int j = 0; j < noOutputs; j++) {
                weights[i][j] = minValue + random.nextDouble() * range;
            }
        }

        return weights;
    }
}