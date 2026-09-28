import numpy as np

@np.vectorize
def relu(x):
    return x if x > 0 else 0

def leaky_relu(leak = 0.1, alpha = 1):
    @np.vectorize
    def func(x):
        return alpha * x if x > 0 else x*leak  # leak is accessed through a 'closure', func.__name__ works here.
    func.__name__ = "leaky_relu" 
    return func

def der_leaky_relu(leak = 0.1, alpha = 1):
    @np.vectorize
    def func(x):
        return alpha if x > 0 else leak
    return func

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@np.vectorize
def round_off(x):
    return 1 if x >= 0.5 else 0

@np.vectorize
def mirror(x):
    return x

@np.vectorize
def der_mirror(x):
    return 1

def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))

@np.vectorize
def der_sigmoid(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))

@np.vectorize
def der_relu(x):
    return 1 if x > 0 else 0

# same interface as obtained thru np.vectorize
def softmax(layer):
    total = np.sum(np.exp(layer))
    return np.exp(layer) / total

class MSE:
    def calculate_loss(self, labels, predictions):
        labels = labels.reshape(predictions.shape)
        loss = (1 / labels.size) * np.sum((labels - predictions)**2)
        return loss
    
    def der_loss(self, label, output):
        return 2 * (output - label)
    
    def error_output_layer(self, layer, label):
        first_term = self.der_loss(label, layer.a_[0][0])
        second_term = layer.der_activate()
        error = np.clip(first_term * second_term, -5, 5)
        return error

class BinaryLoss(MSE):
    def calculate_loss(self, labels, predictions, epsilon = 1e-7):
        total = predictions.size
        labels = labels.reshape(predictions.shape)
        summation = 0

        predictions = np.clip(predictions, epsilon, 1 - epsilon)  # avoiding values of 0 ( => inf loss) and 1 ( => 0 loss)
        iterator = zip(labels, predictions)
        for label, prediction in iterator:
            summation += label * np.log(prediction) + (1 - label) * np.log(1 - prediction)
        return (-1 / total) * summation

    def der_loss(self, label, output, epsilon = 1e-7):  # output belongs to range [0, 1]
        output = np.clip(output, epsilon, 1 - epsilon)
        return ((1 - label)/(1 - output)) - (label / output)

class CategoricalLoss(MSE):
    # labels must be one-hot encoded with drop_first = False; labels & predictions = complete vector of all classes for each instance.
    def calculate_loss(self, labels, predictions, epsilon = 1e-9):
        total = predictions.shape[0]
        labels = labels.reshape(predictions.shape)
        summation = 0

        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        iterator = zip(labels, predictions)
        for label, prediction in iterator:
            summation += np.sum(label * np.log(prediction))
        
        return (-1 / total) * summation
    
    # always use 'softmax' with this
    def error_output_layer(self, layer, label):
        label = label.reshape((label.size, 1))
        return layer.a_ - label
