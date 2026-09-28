import numpy as np

# optimizer will give the trainer the gradients[] matrix to apply to the weights.
class SGD():
    def set_model(self, model):
        self.model = model

    def gradient_weights(self, weight_index):
        return self.model.weights[weight_index].gradients
    
    def gradient_biases(self, layer_index):
        return self.model.layers[layer_index].b_gradients
    
    def current_gradient(self, weight_index):
        return np.matmul(self.model.weights[weight_index].layer_2.del_, np.transpose(self.model.weights[weight_index].layer_1.a_))
    
    def error_output_layer(self, label):
        return self.model.loss.error_output_layer(self.model.layers[-1], label)

    def error_layer(self, this_index, weight_index):  # weights connecting this layer to next layer
        '''
        return np.matmul(
            np.transpose(self.model.weights[weight_index].matrix),
            self.model.layers[this_index+1].del_) * self.model.layers[this_index].der_activate()
        '''
        w = np.transpose(self.model.weights[weight_index].matrix)
        d = self.model.layers[this_index+1].del_
        der = self.model.layers[this_index].der_activate()
        
        #print(f"Max W: {np.max(np.abs(w)):.4e} | Max Delta: {np.max(np.abs(d)):.4e} | Max Der: {np.max(np.abs(der)):.4e}")
        
        return np.matmul(w, d) * der
        
    def on_pass(self):
        # reset gradients
        for weight in self.model.weights:
            weight.gradients = np.zeros((weight.rows, weight.cols))
        
        # reset errors
        for layer in self.model.layers:
            layer.del_ = np.zeros((layer.n_neurons, 1))
            layer.b_gradients = np.zeros((layer.n_neurons, 1))
    
    def update_biases(self, layer_index, l_rate):
        value = l_rate * self.gradient_biases(layer_index)
        self.model.layers[layer_index].b_ -= value
        return value
    
    def update_weights(self, weight_index, l_rate):
        value = l_rate * self.gradient_weights(weight_index)
        self.model.weights[weight_index].matrix -= value
        return value

class Momentum(SGD):
    def __init__(self, beta = 0.9):
        self.beta = beta
    
    def set_model(self, model):
        super().set_model(model)

        # create space for storing momentum terms for every parameter
        self.weights = list(map(lambda weights: np.zeros(weights.matrix.shape), self.model.weights))
        self.biases = list(map(lambda layer: np.zeros(layer.b_.shape), self.model.layers))
    
    # keep momentum upon applying update, not while accumulation along a mini batch
    def update_weights(self, weight_index, l_rate):
        self.weights[weight_index] = self.beta * self.weights[weight_index] - (l_rate * self.gradient_weights(weight_index))  # update momentum term
        self.model.weights[weight_index].matrix += self.weights[weight_index]
        return -self.weights[weight_index]  # because already subtracting in first line

    def update_biases(self, layer_index, l_rate):
        self.biases[layer_index] = self.beta * self.biases[layer_index] - (l_rate * self.gradient_biases(layer_index))  # update momentum term
        self.model.layers[layer_index].b_ += self.biases[layer_index]
        return -self.biases[layer_index]

class RMSProp(Momentum):
    def __init__(self, beta = 0.9):
        self.beta = beta
        self.epsilon = 1e-7

    # use super's set_model()

    def update_weights(self, weight_index, l_rate):
        self.weights[weight_index] = self.beta * self.weights[weight_index] + (1 - self.beta) * np.square(self.gradient_weights(weight_index))
        self.model.weights[weight_index].matrix -= (l_rate * self.gradient_weights(weight_index) / np.sqrt(self.weights[weight_index] + self.epsilon))
        return np.sqrt(self.weights[weight_index] + self.epsilon)

    def update_biases(self, layer_index, l_rate):
        self.biases[layer_index] = self.beta * self.biases[layer_index] + (1 - self.beta) * np.square(self.gradient_biases(layer_index))
        self.model.layers[layer_index].b_ -= (l_rate * self.gradient_biases(layer_index) / np.sqrt(self.biases[layer_index] + self.epsilon))
        return np.sqrt(self.biases[layer_index] + self.epsilon)

# not working as of now
class Adam(SGD):

    def __init__(self, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-7):
        self.weight_mean = list()
        self.weight_std = list()
        self.bias_mean = list()
        self.bias_std = list()
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_power = beta1
        self.beta2_power = beta2
        self.epsilon = epsilon
        self.iteration = 1  # increment this in self.on_pass()
    
    def set_model(self, model):
        super().set_model(model)
        for i in self.model.weights:
            self.weight_mean.append(np.zeros(i.gradients.shape))
            self.weight_std.append(np.zeros(i.gradients.shape))
        for i in self.model.layers:
            self.bias_mean.append(np.zeros(i.b_gradients.shape))
            self.bias_std.append(np.zeros(i.b_gradients.shape))
        
    def reset(self):
        self.iteration = 1
        self.beta1_power = self.beta1
        self.beta2_power = self.beta2

    # time-step increments in on_pass(). but it is used whenever we do update_biases() or update_weights() from the Trainer.
    def gradient_weights(self, weight_index):
        gradients = self.model.weights[weight_index].gradients

        weight_momentum = self.weight_mean[weight_index]
        weight_momentum = self.beta1 * weight_momentum - ((1 - self.beta1) * gradients)
        weight_m_normal = weight_momentum / (1 - self.beta1_power)
        self.weight_mean[weight_index] = weight_m_normal

        weight_scale = self.weight_std[weight_index]
        weight_scale = self.beta2 * weight_scale + ((1 - self.beta2) * np.square(gradients))
        weight_s_normal = weight_scale / (1 - self.beta2_power)
        self.weight_std[weight_index] = weight_s_normal

        return np.divide(weight_m_normal, np.sqrt(weight_s_normal + self.epsilon))
        

    def gradient_biases(self, layer_index):
        gradients = self.model.layers[layer_index].b_gradients

        bias_momentum = self.bias_mean[layer_index]
        bias_momentum = self.beta1 * bias_momentum - ((1 - self.beta1) * gradients)
        bias_m_normal = bias_momentum /  (1 - self.beta1_power)
        self.bias_mean[layer_index] = bias_m_normal

        bias_scale = self.bias_std[layer_index]
        bias_scale = self.beta2 * bias_scale + ((1 - self.beta2) * np.square(gradients))
        bias_s_normal = bias_scale / (1 - self.beta2_power)
        self.bias_std[layer_index] = bias_s_normal

        return np.divide(bias_m_normal, np.sqrt(bias_s_normal + self.epsilon))
    
    
    def on_pass(self):
        super().on_pass()  # yes, reset gradients, [but not iteration?]
        #print(f"adam's on_pass, weight moments = {self.weight_mean}")
        self.iteration += 1
        self.beta1_power *= self.beta1
        self.beta2_power *= self.beta2
        
