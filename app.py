import streamlit as st
import numpy as np
import pandas as pd

# CREATE DIRECTORIES
import os
import shutil

logs_dir = "./logs"
models_dir = "./models"
plots_dir = "./plots"

# Create the folders if they don't exist
if "session_initialized" not in st.session_state:
  for dir in [logs_dir, models_dir, plots_dir]:
      if os.path.exists(dir):
        shutil.rmtree(dir)  # Remove the directory and its contents
      os.makedirs(dir)  # Create the directory
  st.session_state.session_initialized = True  # Mark the session as initialized

# STREAMLIT CONFIG
st.set_page_config(
    page_title="Neural Network Demonstration",
    layout="wide"  # Turns on wide mode to remove huge margins
)

st.title("Neural Network from Scratch", 
  text_alignment="center", 
  icon=":material/network_node:")

@st.cache_data
def load_train():
  X_train, y_train = and_gate_dataset(100, 1)
  dataset = pd.DataFrame(np.hstack((X_train, y_train)), columns=["Input 1", "Input 2", "Output"])
  return dataset

@st.cache_data
def load_test():
  X_test, y_test = and_gate_dataset(50, 2)
  dataset = pd.DataFrame(np.hstack((X_test, y_test)), columns=["Input 1", "Input 2", "Output"])
  return dataset

# DATASET
from nn.dataset_utils import and_gate_dataset

st.header("AND gate dataset")

st.subheader("Training set")
train_data = load_train()
st.dataframe(train_data)

st.subheader("Testing set")
test_data = load_test()
st.dataframe(test_data)

# MODEL
from nn.model_classes import Model, Layer
from nn.functions import BinaryLoss, leaky_relu, der_leaky_relu, sigmoid, der_sigmoid, relu, der_relu

activation_functions = {
    "None": [None, None],
    "ReLU": [relu, der_relu],
    "Leaky ReLU": [leaky_relu(), der_leaky_relu()],
    "Sigmoid": [sigmoid, der_sigmoid]
}

st.header("Model")
st.subheader("Add Layers")
col1, col2, col3, col4 = st.columns(4)

if "model" not in st.session_state:
    st.session_state.model = Model(BinaryLoss(), 1)
    st.session_state.model.add_layer(Layer(2, activation_functions["None"][0], activation_functions["None"][1]))
    st.session_state.model.add_layer(Layer(1, activation_functions["Sigmoid"][0], activation_functions["Sigmoid"][1]))
    st.session_state.model_name = "AND Gate Model"
  
if "activations" not in st.session_state:
    st.session_state.activations = list()
    st.session_state.activations.append("None")
    st.session_state.activations.append("Sigmoid")

model = st.session_state.model
activations = st.session_state.activations

with col1:
  layer_number = st.number_input(f"Layer number", disabled=True, value=len(model.layers))

with col2:
  n_neurons = st.number_input("Number of Neurons", min_value=1, max_value=10, value=2, step=1)

with col3:
  activation_function = st.selectbox("Activation Function", list(activation_functions.keys()))

with col4:
  if st.button("Add Layer", help="Add a new HIDDEN layer to the model (between input and output layers)"):
    model.add_layer(Layer(n_neurons, activation_functions[activation_function][0], activation_functions[activation_function][1]), index=-1)
    activations.insert(-1, activation_function)
    st.success(f"Layer {layer_number} added!")

if st.button("Compile Model"):
  model.compile()
  st.success("Model compiled!")

# Display model layers as table
st.subheader("Model Summary")

layers = pd.DataFrame(columns=["Layer", "Number of Neurons", "Activation Function"])
for i, layer in enumerate(model.layers):
  layers.loc[i] = [i+1, layer.n_neurons, activations[i]]

st.dataframe(layers, hide_index=True)

show_weights_biases = st.checkbox("Show Weights and Biases")

if show_weights_biases:
  st.subheader("Weights and Biases")
  
  # collect weights and biases from model
  biases = []
  weights = []
  for i, layer in enumerate(model.layers):
    biases.append(layer.b_.T)
  for i, weight in enumerate(model.weights):
    weights.append(weight.matrix)

  # display
  for i in range(len(weights)):
    st.write(f"Layer {i+1} Biases:")
    biases[i]
    st.write(f"Next Weights:")
    weights[i]
  st.write(f"Layer {len(weights)+1} Biases:")
  biases[-1]

# PREP PLOTTING
from nn.plotter import Plotter

if 'plotter' not in st.session_state:
  st.session_state.plotter = Plotter()

plotter = st.session_state.plotter

@st.cache_data
def load_history(name = 'batch_size_1.txt'):
  plotter.read_file(f'./logs/{name}')
  return f'./logs/{name}'

@st.cache_data
def plot_gradients(name = 'batch_size_1', points = 700):
  plotter.plot_gradients('./plots', name, points)
  return f'./plots/gradients_{name}.png'

@st.cache_data
def plot_weights(name = 'batch_size_1', points = 700):
  plotter.plot_weights('./plots', name, points)
  return f'./plots/weights_{name}.png'

@st.cache_data
def plot_score(name = 'batch_size_1', points = 700):
  plotter.plot_score('./plots', name, points)
  return f'./plots/score_{name}.png'

@st.cache_data
def plot_predictions(X_train, _dir = './plots', name = 'batch_size_1'):
  plotter.plot_predictions(X_train, _dir, name)
  return f'{_dir}/predictions_{name}.png'

@st.cache_data
def plot_loss_landscape(_trainer, X_train, y_train, _dir = './plots', name = 'batch_size_1'):
  plotter.plot_contours(_trainer, X_train, y_train, _dir, name)
  return f'{_dir}/contours_{name}.png'

# TRAIN
from nn.trainer import Trainer
from nn.optimizers import SGD

st.header("Training")

trainer = Trainer(model, SGD())
X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
y_train = y_train.reshape(y_train.shape[0], 1)  # reshape to column vector

# input fields for training
batch_size = st.number_input("Batch Size (1 - 32)", min_value=1, max_value=32, value=1, step=1)
learning_rate = st.number_input("Learning Rate (0.001 - 1.0)", min_value=0.001, max_value=1.0, value=0.02, step=0.001, format="%.3f")
epochs = st.number_input("Epochs (1 - 500)", min_value=1, max_value=500, value=120, step=1)

if st.button("Train Model"):
  with st.spinner("Training in progress..."):
    trainer.train(X_train, y_train, batch_size, learning_rate, epochs = epochs)
  st.success("Training completed!")

  trainer.save_history('./logs', 'batch_size_1')
  model.save_weights('./models', 'batch_size_1')

  st.cache_data.clear()  # clear all cache to ensure plots are generated with the latest training data

# PLOT
st.header("Training History")

if st.button("Plot History"):
  with st.spinner("Reading log file..."):
    load_history()

  with st.spinner("Plotting gradients..."):
    gradient_path = plot_gradients()
    st.image(gradient_path, caption='Gradients', width='stretch')

  with st.spinner("Plotting weights..."):
    weight_path = plot_weights()
    st.image(weight_path, caption='Weights', width='stretch')

  with st.spinner("Plotting accuracy..."):
    score_path = plot_score()
    st.image(score_path, caption='Accuracy', width='stretch')

  with st.spinner("Plotting outputs..."):
    prediction_path = plot_predictions(X_train)
    st.image(prediction_path, caption='Predictions', width='stretch')

  with st.spinner("Plotting loss landscape (this takes a while)..."):
    loss_landscape_path = plot_loss_landscape(trainer, X_train, y_train, "./plots", "batch_size_1")
    st.image(loss_landscape_path, caption='Loss Landscape', width='stretch')

  st.success("Plots generated!")