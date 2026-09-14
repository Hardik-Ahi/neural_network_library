# Neural Network Library from Scratch
A modular neural network framework with a TensorFlow-like API and custom visualization tooling, built from the ground up using NumPy and Matplotlib.

[Watch Demo on YouTube](https://youtu.be/qJAhmKIAFPU)

### Note:
> Jupyter Notebooks included here don't run anymore due to modifying the library code for **Streamlit** compatibility.

## Quickstart for Windows

1. Install Python 3.9
2. `git clone <repository_url> project_folder`
3. `cd project_folder`
4. `python -m venv my_venv`
5. `my_venv\Scripts\activate`
6. `pip install -r requirements.txt`

## Overview
This project is a fully functional neural network library implemented entirely from scratch using Python, NumPy, and object-oriented design principles.

The goal of the project was not just to train neural networks, but to deeply understand and implement the mathematics, optimization algorithms, and software architecture behind modern deep learning frameworks.

The library provides:

* Layer-by-layer neural network construction
* Forward and backward propagation
* Gradient descent optimization
* Modular optimizer support
* Visualization utilities for training and model behavior
* A TensorFlow-inspired API for ease of use

This project was developed as my college major project and serves as both:

* a proof of concept for neural network training systems, and
* an exploration of software engineering principles applied to machine learning infrastructure.

## Familiar Developer Experience
The API is intentionally designed to resemble popular ML libraries such as TensorFlow/Keras, making it approachable for users already familiar with modern deep learning workflows.

## Why I Built This
I built this project to bridge the gap between:

* using machine learning libraries, and
* understanding how they work internally.

Implementing the entire training pipeline from scratch helped me develop:

* stronger mathematical intuition,
* better debugging skills,
* deeper software engineering experience, and
* a clearer understanding of neural network systems design.

## Example Usage

```python
import pandas as pd
from nn.model_classes import Model, Layer, PolyLayer
from nn.functions import MSE, leaky_relu, der_leaky_relu, sigmoid, der_sigmoid, mirror, der_mirror
from nn.trainer import RegressionTrainer
from nn.plotter import Plotter
from nn.dataset_utils import pca, standardize_data, split_data
from nn.optimizers import SGD, Momentum

dataset = pd.read_csv(r'dataset.csv')
D_train, D_test = split_data(dataset, 0.3, 10)

X_train = D_train.drop(columns = ['target'])
y_train = D_train['target']
X_test = D_test.drop(columns = ['target'])
y_test = D_test['target']

model = Model(MSE(), 5)
model.add_layer(Layer(8))
model.add_layer(PolyLayer(Layer(4, leaky_relu(), der_leaky_relu()), Layer(4, sigmoid, der_sigmoid)))
model.add_layer(PolyLayer(Layer(2, leaky_relu(0.01, 0.2), der_leaky_relu(0.01, 0.2)), Layer(2, leaky_relu(0.2, 0.01), der_leaky_relu(0.2, 0.01))))
model.add_layer(Layer(1, mirror, der_mirror))
model.compile()

trainer = RegressionTrainer(model, SGD())
trainer.train(X_train, y_train, 4, 0.02, 25)
trainer.save_history('./logs', 'regression')

plotter = Plotter()
plotter.read_file(r'logs\regression.txt')
plotter.plot_gradients("./plots", "regression", 700)
plotter.plot_weights("./plots", "regression", 700)
plotter.plot_score("./plots", "regression", 700, False)

_ = trainer.predict(X_test, y_test)

axis = pca(D_train, "target", 1)
plotter.plot_regression(axis, D_train["target"].to_numpy(), "./plots", "regression", x_label = "PCA Component-1")
plotter.plot_contours(trainer, X_train, y_train, "./plots", "regression", magnitude = 0.2)
```

## Visualizations (Plots)

### Accuracy / Loss Curves
![](sample_plots/accuracy_batch_size_16.png)

### Gradients Per Update
![](sample_plots/gradients_batch_size_1.png)

### Binary Classification Predictions (PCA Plot)
![](sample_plots/classification_dataset_classification.png)

### Multiple Linear Regression Predictions (PCA Plot)
![](sample_plots/regression_dataset_regression.png)

### Loss Landscape (Contour Plot)
![](sample_plots/contours_batch_size_1.png)
