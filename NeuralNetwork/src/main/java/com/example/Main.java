package com.example;

import com.example.activations.ReLU;
import com.example.activations.Sigmoid;
import com.example.core.NetworkTrainer;
import com.example.core.NeuralNetworkEngine;
import com.example.init.He;
import com.example.init.Xavier;
import com.example.layers.DenseLayer;
import com.example.loss.CrossEntropy;
import com.example.utils.CSVDataReader;
import com.example.utils.DataSplitter;


public class Main {
    public static void main(String[] args) {
       try {
           System.out.println("Loading data from CSV...");

           

           CSVDataReader reader = new CSVDataReader("D:\\projects\\demo\\resources\\processed_data.csv",true);
           reader.loadData();
           reader.printSummary();

           double[][] data = reader.getDataAsDoubleArray();
           int numSamples = data.length;
           int numFeatures = data[0].length;

           double[][] X = new double[numSamples][numFeatures];
           double[][] y = new double[numSamples][1];


           for(int i = 0; i < numSamples; i++){
            int featureIdx = 0;
            for(int j = 0; j < numFeatures; j++){
                if (j == 9) {
                    y[i][0] = data[i][j];
                } else {
                    X[i][featureIdx] = data[i][j];
                    featureIdx++;
                }
            }
        }

           System.out.println("Data loaded");

           System.out.println("Rows = " + numSamples);
           System.out.println("Features = " + numFeatures);
           System.out.println("============================================================");
    
           System.out.println("Splitting data into train, test");
           DataSplitter splitter = new DataSplitter(42);

           double[][][] split = splitter.trainTestSplit(X, y, 0.2);
           double[][] X_train = split[0];
           double[][] X_test = split[1];
           double[][] y_train = split[2];
           double[][] y_test = split[3];

           System.out.println("Training samples: " + X_train.length);
           System.out.println("Test samples: " + X_test.length);
           System.out.println("============================================================");

           System.out.println("Building neural network");
           NeuralNetworkEngine engine = new NeuralNetworkEngine();

            // engine.addLayer(new DenseLayer(numFeatures, 256, new RandomUniform(-Math.sqrt(6.0/numFeatures), Math.sqrt(6.0/numFeatures), 42), new ReLU()));
            // engine.addLayer(new DenseLayer(256, 128, new RandomUniform(-Math.sqrt(6.0/256), Math.sqrt(6.0/256), 42), new ReLU()));
            // engine.addLayer(new DenseLayer(128, 64, new RandomUniform(-Math.sqrt(6.0/128), Math.sqrt(6.0/128), 42), new ReLU()));
            // engine.addLayer(new DenseLayer(64, 32, new RandomUniform(-Math.sqrt(6.0/64), Math.sqrt(6.0/64), 42), new ReLU()));
            // engine.addLayer(new DenseLayer(32, 10, new RandomUniform(-Math.sqrt(6.0/32), Math.sqrt(6.0/32), 42), new ReLU()));
            // engine.addLayer(new DenseLayer(10, 1, new RandomUniform(-Math.sqrt(6.0/11), Math.sqrt(6.0/11), 42), new Sigmoid()));

            engine.addLayer(new DenseLayer(numFeatures, 256, new He(42), new ReLU()));
            engine.addLayer(new DenseLayer(256, 128, new He(42), new ReLU()));
            engine.addLayer(new DenseLayer(128, 64, new He(42), new ReLU()));
            engine.addLayer(new DenseLayer(64, 32, new He(42), new ReLU()));
            engine.addLayer(new DenseLayer(32, 10, new He(42), new ReLU()));
            engine.addLayer(new DenseLayer(10, 1, new Xavier(), new Sigmoid()));

           engine.printSummary();
           System.out.println("============================================================");

           System.out.println("Config the neural network");
           NetworkTrainer trainer = new NetworkTrainer(engine, 42);
           trainer.setLossFunction(new CrossEntropy());
           trainer.setLearningRate(0.1);
           trainer.setEpochs(300);
           trainer.setBatchSize(32);
           trainer.setVerbose(true);

           System.out.println("============================================================");
           System.out.println("TRAINING NEURAL NETWORK");
           System.out.println("============================================================");

           trainer.train(X_train, y_train);

           System.out.println("============================================================");
           System.out.println("Evaluation");
           System.out.println("============================================================");

           
           double testLoss = engine.evaluate(X_test, y_test, trainer.getLossFunction());
           System.out.println("Test Loss: " + String.format("%.6f", testLoss));

           double accuracy = calculateAccuracy(engine, X_test, y_test);
           System.out.println("Test Accuracy: " + String.format("%.2f%%", accuracy));

       } catch (Exception e) {
        System.err.println("Error" + e.getMessage());
            e.printStackTrace();
       }
        
    }

     private static double calculateAccuracy(NeuralNetworkEngine engine, double[][] X_test, double[][] y_test) {
        double[][] predictions = engine.predict(X_test);
        int correct = 0;
        
        for (int i = 0; i < predictions.length; i++) {
            double predicted = predictions[i][0] > 0.5 ? 1.0 : 0.0;
            double actual = y_test[i][0];
            
            if (Math.abs(predicted - actual) < 0.01) {
                correct++;
            }
        }
        
        return (double) correct / predictions.length * 100.0;
    }
}