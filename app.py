import streamlit as st
import numpy as np
import pandas as pd

# STREAMLIT CONFIG
st.set_page_config(
    page_title="Neural Network Demonstration",
    layout="wide"  # Turns on wide mode to remove huge margins
)

st.title("Neural Network from Scratch", 
  text_alignment="center", 
  icon=":material/network_node:")

if st.button("Reset Current Session", type="primary"):
    for key in st.session_state.keys():
        del st.session_state[key]

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
from nn.dataset_utils import and_gate_dataset, standardize_data, split_classes, split_data

st.header("Dataset")

dataset_option = st.selectbox("Select Dataset", ["AND Gate (Built-in)", "Upload a Dataset"])

if dataset_option == "Upload a Dataset":
  uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
  if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.dataframe(dataset)
    st.subheader("Preprocess Dataset")

    # 1. Select the target column and its type
    target_column = st.selectbox("Select the target column", dataset.columns)
    target_type = st.selectbox("Select the target type", ["regression", "classification"])

    # 2. select columns to one-hot encode
    one_hot_columns = st.multiselect("Select columns to one-hot encode", dataset.columns)
    if st.button("Apply One-Hot Encoding"):
      if one_hot_columns:
          dataset = pd.get_dummies(dataset, columns=one_hot_columns, drop_first=True, dtype=int)
          st.success("One-hot encoding applied!")
    
    # 3. split into train, test sets
    if target_type == "classification":
      train_set, test_set = split_classes(dataset, target_column)  # preserves class balance
    elif target_type == "regression":
      train_set, test_set = split_data(dataset)

    # 4. select columns to standardize, maybe including target
    standardize_columns = st.multiselect("Select columns to standardize", dataset.columns)
    if st.button("Apply Standardization"):
      if standardize_columns:
        X_means, X_stds = standardize_data(train_set[standardize_columns])
        standardize_data(test_set[standardize_columns], from_means=X_means, from_stds=X_stds)
        st.success("Standardization applied!")

    # 5. extract features and labels for train and test sets
      X_train, y_train = train_set.drop(columns=[target_column]), train_set[target_column]
      X_test, y_test = test_set.drop(columns=[target_column]), test_set[target_column]
  
elif dataset_option == "AND Gate (Built-in)":
  st.subheader("Training set")
  train_data = load_train()
  st.dataframe(train_data)

  st.subheader("Testing set")
  test_data = load_test()
  st.dataframe(test_data)

  # Extract features and targets for train and test sets
  X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
  y_train = y_train.reshape(y_train.shape[0], 1)  # reshape to column vector

  X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values
  y_test = y_test.reshape(y_test.shape[0], 1)  # reshape to column vector

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
st.subheader("Add Hidden Layers")
col1, col2, col3, col4 = st.columns(4)

if "model" not in st.session_state:
    st.session_state.model = Model(BinaryLoss(), 1)
    st.session_state.model.add_layer(Layer(2, activation_functions["None"][0], activation_functions["None"][1]))
    st.session_state.model.add_layer(Layer(1, activation_functions["Sigmoid"][0], activation_functions["Sigmoid"][1]))
    st.session_state.model_compiled = False
  
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

# Display model layers as table
st.subheader("Model Summary")

layers = pd.DataFrame(columns=["Layer", "Number of Neurons", "Activation Function"])
for i, layer in enumerate(model.layers):
  layers.loc[i] = [i+1, layer.n_neurons, activations[i]]

st.dataframe(layers, hide_index=True)

if st.button("Compile Model"):
  model.compile()
  st.session_state.model_compiled = True
  st.success("Model compiled!")

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
    st.write(biases[i])
    st.write(f"Next Weights:")
    st.write(weights[i])
  st.write(f"Layer {len(weights)+1} Biases:")
  st.write(biases[-1])

# PREP PLOTTING
from nn.plotter import Plotter
from nn.trainer import Logger
import io

if 'plotter' not in st.session_state:
  st.session_state.plotter = Plotter()

if 'data' not in st.session_state:
  st.session_state.data = None

plotter = st.session_state.plotter

@st.cache_data
def load_history(string):  # returns JSON (Python) object from string - serializable
  return Logger.load_data(string)

@st.cache_data
def plot_gradients(string):
  gradients = plotter.plot_gradients(st.session_state.data, n_points=700)
  buf = io.BytesIO()
  gradients.savefig(buf, format="png", bbox_inches="tight")
  buf.seek(0)
  image_bytes = buf.getvalue()
  return image_bytes

@st.cache_data
def plot_weights(string):
  weights = plotter.plot_weights(st.session_state.data, n_points=700)
  buf = io.BytesIO()
  weights.savefig(buf, format="png", bbox_inches="tight")
  buf.seek(0)
  image_bytes = buf.getvalue()
  return image_bytes

@st.cache_data
def plot_score(string):
  score = plotter.plot_score(st.session_state.data, n_points=700)
  buf = io.BytesIO()
  score.savefig(buf, format="png", bbox_inches="tight")
  buf.seek(0)
  image_bytes = buf.getvalue()
  return image_bytes

@st.cache_data
def plot_predictions(string, _X_train):  # _ underscore means skip hashing on this key
  predictions = plotter.plot_predictions(st.session_state.data, _X_train)
  buf = io.BytesIO()
  predictions.savefig(buf, format="png", bbox_inches="tight")
  buf.seek(0)
  image_bytes = buf.getvalue()
  return image_bytes

@st.cache_data
def plot_loss_landscape(string, _trainer, _X_train, _y_train):
  contours = plotter.plot_contours(st.session_state.data, _trainer, _X_train, _y_train)
  buf = io.BytesIO()
  contours.savefig(buf, format="png", bbox_inches="tight")
  buf.seek(0)
  image_bytes = buf.getvalue()
  return image_bytes

# TRAIN
from nn.trainer import Trainer
from nn.optimizers import SGD

st.header("Training")

trainer = Trainer(model, SGD())

# input fields for training
batch_size = st.number_input("Batch Size (1 - 32)", min_value=1, max_value=32, value=1, step=1)
learning_rate = st.number_input("Learning Rate (0.001 - 1.0)", min_value=0.001, max_value=1.0, value=0.02, step=0.001, format="%.3f")
epochs = st.number_input("Epochs (1 - 500)", min_value=1, max_value=500, value=120, step=1)

if 'history' not in st.session_state:
  st.session_state.history = None

if st.button("Train Model"):
  if not st.session_state.model_compiled:
    st.error("Please compile the model first.")
    st.stop()

  with st.spinner("Training in progress..."):
    trainer.train(X_train, y_train, batch_size, learning_rate, epochs = epochs)

  st.success("Training completed!")
  st.session_state.history = trainer.save_history()

# PLOT
st.header("Training History")

if st.button("Plot History"):
  if st.session_state.history is None:
    st.error("No training history found. Please train the model first.")
    st.stop()

  with st.spinner("Reading training history..."):
    st.session_state.data = load_history(st.session_state.history)

  with st.spinner("Plotting gradients..."):
    gradient_path = plot_gradients(st.session_state.history)
    st.subheader("Gradients")
    st.image(gradient_path, width='stretch')

  with st.spinner("Plotting weights..."):
    weight_path = plot_weights(st.session_state.history)
    st.subheader("Weights")
    st.image(weight_path, width='stretch')

  with st.spinner("Plotting accuracy..."):
    score_path = plot_score(st.session_state.history)
    st.subheader("Accuracy")
    st.image(score_path, width='stretch')

  with st.spinner("Plotting outputs..."):
    prediction_path = plot_predictions(st.session_state.history, X_train)
    st.subheader("Predictions")
    st.image(prediction_path, width='stretch')

  with st.spinner("Plotting loss landscape (this takes a while)..."):
    loss_landscape_path = plot_loss_landscape(st.session_state.history, trainer, X_train, y_train)
    st.subheader("Loss Landscape")
    st.image(loss_landscape_path, width='stretch')

  st.success("Plots generated!")