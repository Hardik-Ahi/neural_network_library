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

def clear_session():
  for key in st.session_state.keys():
    del st.session_state[key]

if st.button("Reset Current Session", type="primary"):
  clear_session()

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
from nn.model_classes import Model
from nn.functions import MSE, BinaryLoss

st.header("Dataset")
if 'target_type' not in st.session_state:
  st.session_state.target_type = None
  st.session_state.target_name = None

if 'train_set' not in st.session_state:
  st.session_state.train_set = None
  st.session_state.test_set = None
  st.session_state.X_y_train = []
  st.session_state.X_y_test = []

dataset_option = st.selectbox("Select Dataset", ["AND Gate (Built-in)", "Upload a Dataset"])

if dataset_option == "Upload a Dataset":
  uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
  if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.dataframe(dataset)
    st.subheader("Preprocess Dataset")

    with st.form("preprocess_form"):  # separate re-runs based UI from logic
      # 1. Select the target column and its type
      target_column = st.selectbox("Select the target column", dataset.columns)
      target_type = st.selectbox("Select the target type", ["regression", "classification"])

      # 2. select columns to one-hot encode
      one_hot_columns = st.multiselect("Select columns to one-hot encode", dataset.columns)

      # 3. select columns to standardize, maybe including target
      standardize_columns = st.multiselect("Select columns to standardize", dataset.columns)

      # 4. submit button
      submitted = st.form_submit_button("Apply Preprocessing")

    if submitted:
      # 0. Store target type for further use
      st.session_state.target_type = target_type
      st.session_state.target_name = target_column

      # 0.5 Remove NA rows - silently
      dataset = dataset.dropna(ignore_index=True)

      # 1. Apply one-hot encoding
      if one_hot_columns:
          dataset = pd.get_dummies(dataset, columns=one_hot_columns, drop_first=True, dtype=int)
          st.success("One-hot encoding applied!")

      # 2. Split into train and test sets
      if target_type == "classification":
        print("using classification target")
        train_set, test_set = split_classes(dataset, target_column)  # preserves class balance
        st.session_state.model = Model(BinaryLoss(), 1)
      elif target_type == "regression":
        print("using regression target")
        train_set, test_set = split_data(dataset)
        st.session_state.model = Model(MSE(), 1)
      
      st.session_state.model_compiled = False
      
      # 2.5 Store train and test sets in session state
      st.session_state.train_set = train_set
      st.session_state.test_set = test_set

      # 3. Standardize selected columns
      if standardize_columns:
        X_means, X_stds = standardize_data(train_set, standardize_columns)
        standardize_data(test_set, standardize_columns, from_means=X_means, from_stds=X_stds)
        st.success("Standardization applied!")

      # 4. extract features and labels for train and test sets
      X_train, y_train = train_set.drop(columns=[target_column]), train_set[target_column]
      st.session_state.X_y_train = [X_train.to_numpy(), y_train.to_numpy().reshape(-1, 1)]
      X_test, y_test = test_set.drop(columns=[target_column]), test_set[target_column]
      st.session_state.X_y_test = [X_test.to_numpy(), y_test.to_numpy().reshape(-1, 1)]
  
elif dataset_option == "AND Gate (Built-in)":
  st.session_state.target_type = "classification"
  st.session_state.target_name = "Output"
  st.session_state.train_set = load_train()
  st.session_state.test_set = load_test()

  # Extract features and targets for train and test sets
  X_train, y_train = st.session_state.train_set.iloc[:, :-1].values, st.session_state.train_set.iloc[:, -1].values
  y_train = y_train.reshape(y_train.shape[0], 1)  # reshape to column vector
  st.session_state.X_y_train = [X_train, y_train]

  X_test, y_test = st.session_state.test_set.iloc[:, :-1].values, st.session_state.test_set.iloc[:, -1].values
  y_test = y_test.reshape(y_test.shape[0], 1)  # reshape to column vector
  st.session_state.X_y_test = [X_test, y_test]

  st.session_state.model = Model(BinaryLoss(), 1)
  st.session_state.model_compiled = False

# show train and test sets globally
if st.session_state.train_set is None:
  st.stop()

st.subheader("Training Set")
train1, train2 = st.columns(2)
with train1:
  st.subheader("Features")
  st.dataframe(st.session_state.train_set.drop(columns=[st.session_state.target_name]))
with train2:
  st.subheader("Target")
  st.dataframe(st.session_state.train_set[[st.session_state.target_name]])

st.subheader("Testing Set")
test1, test2 = st.columns(2)
with test1:
  st.subheader("Features")
  st.dataframe(st.session_state.test_set.drop(columns=[st.session_state.target_name]))
with test2:
  st.subheader("Target")
  st.dataframe(st.session_state.test_set[[st.session_state.target_name]])

# MODEL
from nn.model_classes import Model, Layer
from nn.functions import BinaryLoss, MSE, leaky_relu, der_leaky_relu, sigmoid, der_sigmoid, relu, der_relu, mirror, der_mirror

activation_functions = {
    "None": [None, None],
    "ReLU": [relu, der_relu],
    "Leaky ReLU": [leaky_relu(), der_leaky_relu()],
    "Sigmoid": [sigmoid, der_sigmoid],
    "Linear": [mirror, der_mirror]
}

st.header("Model")
st.subheader("Add Layers")
  
if "activations" not in st.session_state:
    st.session_state.activations = list()

model = st.session_state.model
activations = st.session_state.activations

with st.form("add layers"):
  col1, col2, col3, col4 = st.columns(4)
  with col1:
    layer_number = st.number_input(f"Layer number", disabled=True, value=len(model.layers)+1)

  with col2:
    n_neurons = st.number_input("Number of Neurons", min_value=1, max_value=10, value=2, step=1)

  with col3:
    activation_function = st.selectbox("Activation Function", list(activation_functions.keys()))

  with col4:
    submitted = st.form_submit_button("Add Layer", help="Add a new layer to the model")

if submitted:
  model.add_layer(Layer(n_neurons, activation_functions[activation_function][0], activation_functions[activation_function][1]))
  activations.append(activation_function)
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
  print("Any NaNs in X?", np.isnan(st.session_state.X_y_train[0]).any())
  print("Any NaNs in y?", np.isnan(st.session_state.X_y_train[1]).any())
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
from nn.dataset_utils import pca
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
  score = plotter.plot_score(st.session_state.data, n_points=700, confusion_matrix=(st.session_state.target_type == "classification"))
  buf = io.BytesIO()
  score.savefig(buf, format="png", bbox_inches="tight")
  buf.seek(0)
  image_bytes = buf.getvalue()
  return image_bytes

@st.cache_data
def plot_and_gate_predictions(string, _X_train):  # _ underscore means skip hashing on this key
  predictions = plotter.plot_predictions(st.session_state.data, _X_train)
  buf = io.BytesIO()
  predictions.savefig(buf, format="png", bbox_inches="tight")
  buf.seek(0)
  image_bytes = buf.getvalue()
  return image_bytes

@st.cache_data
def plot_regression(string):
  axis = pca(st.session_state.X_y_train[0], n_components=1)
  regression = plotter.plot_regression(st.session_state.data, axis, st.session_state.X_y_train[1])
  buf = io.BytesIO()
  regression.savefig(buf, format="png", bbox_inches="tight")
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
from nn.trainer import Trainer, RegressionTrainer
from nn.optimizers import SGD

st.header("Training")

if st.session_state.target_type == "classification":
  trainer = Trainer(model, SGD())
elif st.session_state.target_type == "regression":
  trainer = RegressionTrainer(model, SGD())

# input fields for training
with st.form("training_form"):
  batch_size = st.number_input("Batch Size (1 - 32)", min_value=1, max_value=32, value=1, step=1)
  learning_rate = st.number_input("Learning Rate (0.001 - 1.0)", min_value=0.001, max_value=1.0, value=0.02, step=0.001, format="%.3f")
  epochs = st.number_input("Epochs (1 - 500)", min_value=1, max_value=500, value=25, step=1)

  submitted = st.form_submit_button("Train Model")

if submitted:
  if not st.session_state.model_compiled:
    st.error("Please compile the model first.")
    st.stop()

  with st.spinner("Training in progress..."):
    if st.session_state.X_y_train is None:
      st.error("No training data found. Please load the dataset first.")
      st.stop()
    X_train, y_train = st.session_state.X_y_train
    trainer.train(X_train, y_train, batch_size, learning_rate, epochs = epochs)

  st.success("Training completed!")
  st.session_state.history = trainer.save_history()

if 'history' not in st.session_state:
  st.session_state.history = None

# PLOT
st.header("Training History")

if st.button("Plot History"):
  if st.session_state.history is None:
    st.error("No training history found. Please train the model first.")
    st.stop()

  X_train, y_train = st.session_state.X_y_train

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
    if st.session_state.target_type == 'regression':
      predictions = plot_regression(st.session_state.history)
    elif st.session_state.target_type == 'classification':
      predictions = plot_and_gate_predictions(st.session_state.history, X_train)  # just AND gate for now
    st.subheader("Predictions")
    st.image(predictions, width='stretch')

  with st.spinner("Plotting loss landscape (this takes a while)..."):
    loss_landscape_path = plot_loss_landscape(st.session_state.history, trainer, X_train, y_train)
    st.subheader("Loss Landscape")
    st.image(loss_landscape_path, width='stretch')

  st.success("Plots generated!")