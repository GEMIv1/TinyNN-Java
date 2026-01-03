
**TinyNN-Java is a from-scratch Java implementation of feedforward neural networks. It includes dense layers, common activations (ReLU, Sigmoid, Tanh), loss functions (MSE, MAE, CrossEntropy), weight initializers (He, Xavier), and CSV utilities for training datasets—designed for learning and experimentation.** 

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Data & Preprocessing](#data--preprocessing)
- [Requirements](#requirements)
- [Build & Run](#build--run)
- [Configuration & Usage](#configuration--usage)
- [Extending the Project](#extending-the-project)


---

## Overview
This repository contains a lightweight, from-scratch implementation of a feedforward neural network and simple training loop implemented in plain Java. It was inspired by an earlier Python notebook implementation ([impl.ipynb](https://github.com/GEMIv1/ML-Projects/blob/main/Neural_Network_Scratch/impl.ipynb)) — an exploratory, non-library notebook. It provides a full library for building and training feedforward neural networks (including both forward and backward propagation used during training). It's intended for learning and experimentation with core ideas such as:
- Layers and activations
- Weight initializers (He, Xavier, RandomUniform)
- Loss functions (Cross-Entropy, MSE, MAE)
- Mini-batch training and evaluation

This repository also includes a simple test/demo: the `Main` class loads a CSV dataset, splits it, builds a network, trains using backpropagation, and evaluates performance.


## Features
- Dense (fully-connected) layers
- Implements both forward and backward propagation (used during training/backpropagation)
- Common activation functions: ReLU, Sigmoid, Tanh, Linear
- Weight initializers: He, Xavier, RandomUniform
- Loss implementations: Cross-Entropy, Mean Squared Error (MSE), Mean Absolute Error (MAE)
- Simple CSV reader and train/test splitting utilities


## Project Structure
- `src/main/java/com/example/activations/` — activation functions
- `src/main/java/com/example/init/` — weight initializers
- `src/main/java/com/example/layers/` — layer interfaces and implementations
- `src/main/java/com/example/loss/` — loss functions
- `src/main/java/com/example/core/` — `NeuralNetworkEngine`, `NetworkTrainer`
- `src/main/java/com/example/utils/` — `CSVDataReader`, `DataSplitter`
- `resources/` — data and notebook for preprocessing (see below)


## Data & Preprocessing
- `resources/heart_disease.csv` — original dataset (raw)
- `resources/cleaning_data.ipynb` — Jupyter notebook used to clean and prepare data
- `resources/processed_data.csv` — processed dataset used by `Main`


## Requirements
- Java 17 (project is compiled for Java 17)
- Maven (for building)

(See `pom.xml` where `<maven.compiler.source>` and `<maven.compiler.target>` are set to 17)


## Build & Run
1. Build the project:

```bash
mvn -DskipTests package
```

2. Run the project (one of these options):
- Run from compiled classes:

```bash
java -cp target/classes com.example.Main
```

- Or run the project directly from your IDE (recommended for development)

- If you prefer a packaged jar (note: manifest may not set `Main-Class`):

```bash
java -cp target/demo-1.0-SNAPSHOT.jar com.example.Main
```


## Configuration & Usage
`Main.java` demonstrates a complete example:
- Loads dataset (CSV), splits into train/test
- Builds a network (configurable layers/initializers/activations)
- Train using `NetworkTrainer` with settings:
  - Loss: `CrossEntropy`
  - Learning rate: `0.1`
  - Epochs: `300`
  - Batch size: `32`
  - Seed: `42`

To customize behavior:
- Modify the network architecture by editing the `engine.addLayer(...)` calls in `Main`
- Change initializers by using `new He(seed)`, `new Xavier()`, or `new RandomUniform(...)`
- Use different loss functions by calling `trainer.setLossFunction(...)`

## Acknowledgements
This project was inspired by an earlier Python notebook implementation: [impl.ipynb](https://github.com/GEMIv1/ML-Projects/blob/main/Neural_Network_Scratch/impl.ipynb). The original work was an exploratory notebook (not a packaged library).
---

