import numpy as np
from numpy.random import default_rng
from nn.dataset_utils import get_minibatch
from nn.functions import round_off
import json, os, time

class Trainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.optimizer.set_model(self.model)
        self.logger = Logger()
    
    def error_output_layer(self, label):
        value = self.optimizer.error_output_layer(label)
        self.model.layers[-1].b_gradients += value / self.batch_size
        self.model.layers[-1].del_ = value

    def error_layer(self, this_index, weight_index):
        value = self.optimizer.error_layer(this_index, weight_index)
        self.model.layers[this_index].b_gradients += value / self.batch_size
        self.model.layers[this_index].del_ = value

    def backward_pass(self, label):  # single instance
        self.error_output_layer(label)
        for i in range(len(self.model.layers)-2, 0, -1):  # except the input layer
            self.error_layer(i, i)
        
        for i in range(len(self.model.weights)):
            self.current_gradient(i)
    
    def current_gradient(self, weight_index):
        self.model.weights[weight_index].gradients += self.optimizer.current_gradient(weight_index) / self.batch_size
    
    def update_biases(self, layer_index):
        return self.optimizer.update_biases(layer_index, self.learning_rate)  # final, applied value (includes scaling by learning rate)
    
    def update_weights(self, weight_index):
        return self.optimizer.update_weights(weight_index, self.learning_rate)
    
    def forward_pass(self, input_data):
        # input_data shape = (1, n_cols)
        self.model.layers[0].a_ = input_data.reshape((input_data.shape[-1], 1))
        for i in range(1, len(self.model.layers)):
            self.model.layers[i].z_ = np.matmul(self.model.weights[i-1].matrix, self.model.layers[i-1].a_) + self.model.layers[i].b_
            self.model.layers[i].activate()
        
    def save_history(self, dir, name = None):
        self.logger.write_log(dir, name)
    
    def train(self, features, targets, batch_size = None, learning_rate = 0.01, epochs = 1, log_epochs = None):
        self.minibatch_size = batch_size  # None means batch GD. for stochastic, specify '1'.
        self.learning_rate = learning_rate
        if self.minibatch_size is None:
            self.minibatch_size = features.shape[0]
        if log_epochs is None:
            log_epochs = epochs

        targets.reshape((targets.size,))
        self.logger.set_limit(log_epochs)
        self.logger.log_init(self.model.weights, self.model.layers)
        for epoch in range(epochs):
            print(f"epoch-{epoch}:")
            self.logger.log_epoch(epoch)
            start_index = 0
            batch = 0
            while start_index < features.shape[0]:
                real_features, real_targets = get_minibatch(features, targets, self.minibatch_size, start_index)
                real_targets = real_targets.reshape((real_targets.size,))
                self.batch_size = real_features.shape[0]  # real batch size for correctly normalizing gradients for accumulation
                for i in range(real_features.shape[0]):
                    self.forward_pass(real_features[i])
                    self.backward_pass(real_targets[i])

                self.logger.log_updates_init(batch)
                weights = list()
                bias = list()
                
                for j in range(1, len(self.model.layers)):
                    bias.append(self.update_biases(j))
                for j in range(len(self.model.weights)):
                    weights.append(self.update_weights(j))

                self.logger.log_update_weights(0, weights[0])
                self.logger.log_update_weights(len(weights)-1, weights[-1])
                self.logger.log_update_bias(1, bias[0])
                self.logger.log_update_bias(len(bias), bias[-1])

                self.logger.log_updates_finish(self.model.weights, self.model.layers)

                self.optimizer.on_pass()
                start_index += self.batch_size
                batch += 1
            self.logger.log_n_updates(batch)
            self.predict(features, targets, for_plot = True)
    
    def confusion_matrix(self, labels, predictions):
        predictions = predictions.reshape((predictions.size, 1))
        labels = labels.reshape((labels.size, 1))
        stack = np.hstack((labels, predictions))
        matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

        for i in range(stack.shape[0]):
            if stack[i][0] == 1:
                if stack[i][1] == 1:
                    matrix['tp'] += 1
                else:
                    matrix['fn'] += 1
            else:
                if stack[i][1] == 1:
                    matrix['fp'] += 1
                else:
                    matrix['tn'] += 1
        return matrix
    
    def accuracy(self, labels, predictions):
        confusion_matrix = self.confusion_matrix(labels, predictions)
        total = 0
        for i in confusion_matrix.values():
            total += i
        return (confusion_matrix['tp'] + confusion_matrix['tn']) / total
    
    def predict(self, features, targets, output_incorrect = False, for_plot = False, quiet = False):
        targets = targets.reshape((targets.size,))
        predictions = list()

        for i in range(features.shape[0]):
            self.forward_pass(features[i])
            predictions.append(self.model.layers[-1].a_[0][0])
        
        predictions = np.asarray(predictions)
        loss = self.model.loss.calculate_loss(targets, predictions.reshape((predictions.size,)))
        predictions = round_off(predictions)
        score = self.accuracy(targets, predictions)
        matrix = self.confusion_matrix(targets, round_off(predictions))

        if not quiet:
            print("Loss:", loss)
            print("Accuracy:", score)
            print("Confusion matrix:", matrix)

        if for_plot:
            self.logger.log_score(loss, score, matrix)
            self.logger.log_predictions(predictions)
        if output_incorrect:
            exists = False
            for i in range(targets.size):
                if targets[i] != predictions[i]:
                    if exists == False:
                        exists = True
                        print("incorrect predictions:")
                    print(f'input = {features[i]}, label = {targets[i]}, prediction = {predictions[i]}')
        return predictions, loss, score

class RegressionTrainer(Trainer):
    def predict(self, features, targets, for_plot = False, quiet = False):
        targets = targets.reshape((targets.size,))
        predictions = list()

        for i in range(features.shape[0]):
            self.forward_pass(features[i])
            predictions.append(self.model.layers[-1].a_[0][0])

        predictions = np.asarray(predictions)
        loss = self.model.loss.calculate_loss(targets, predictions)

        # R^2 score
        residuals = np.sum((targets - predictions)**2)
        means = np.sum((targets - np.mean(targets))**2)
        score = 1 - (residuals / means)

        if not quiet:
            print("Loss:", loss)
            print("R2 score:", score)

        if for_plot:
            self.logger.log_score(loss, score)
            self.logger.log_predictions(predictions)
        return predictions, loss, score

class Logger:
    def __init__(self):
        self.reset()

    def reset(self):
        self.object = dict()
        self.epoch = 0
        self.batch = 0
        self.fp = 0
        self.bp = 0
        self.update = 0
    
    def set_limit(self, limit):
        self.limit = limit-1
        self.stop = False
    
    def log_init(self, model_weights, model_layers):
        self.object['init'] = dict()
        dict_ = self.object['init']
        dict_['weights'] = dict()
        dict_['bias'] = dict()

        dict_['weights'][f'weights-{0}'] = model_weights[0].matrix.tolist()
        dict_['weights'][f'weights-{len(model_weights)-1}'] = model_weights[-1].matrix.tolist()
        self.object['n_weights'] = [0, len(model_weights)-1]
        
        dict_['bias'][f'bias-{1}'] = model_layers[1].b_.tolist()
        dict_['bias'][f'bias-{len(model_weights)}'] = model_layers[-1].b_.tolist()
    
    def log_epoch(self, epoch):
        if epoch > self.limit:
            self.stop = True
            return
        self.epoch = epoch
        self.object[f'epoch-{epoch}'] = dict()
    
    def log_n_updates(self, batch):
        if self.stop:
            return
        self.object[f'epoch-{self.epoch}']['n-updates'] = batch
    
    def log_updates_init(self, update):
        if self.stop:
            return
        ref = self.object[f'epoch-{self.epoch}']
        ref[f'update-{update}'] = dict()
        self.update = update

    def log_update_weights(self, index, weights_gradient):
        if self.stop:
            return
        self.object[f'epoch-{self.epoch}'][f'update-{self.update}'][f'weights-gradient-{index}'] = weights_gradient.tolist()
    
    def log_update_bias(self, index, bias_gradient):
        if self.stop:
            return
        self.object[f'epoch-{self.epoch}'][f'update-{self.update}'][f'bias-gradient-{index}'] = bias_gradient.tolist()
    
    def log_updates_finish(self, model_weights, model_layers):
        if self.stop:
            return
        ref = self.object[f'epoch-{self.epoch}'][f'update-{self.update}']

        for i in self.object['n_weights']:
            ref[f'weights-{i}'] = model_weights[i].matrix.tolist()
        for i in self.object['n_weights']:
            i += 1
            ref[f'bias-{i}'] = model_layers[i].b_.tolist()
    
    def log_predictions(self, predictions):
        predictions = predictions[:200]  # to limit size of log file
        ref = self.object[f'epoch-{self.epoch}']
        ref['predictions'] = predictions.tolist()
    
    def log_score(self, loss, score, confusion_matrix = None):
        if self.stop:
            return
        ref = self.object[f'epoch-{self.epoch}']
        ref['score'] = score
        ref['loss'] = loss
        if confusion_matrix is not None:
            ref['confusion-matrix'] = confusion_matrix
    
    '''
    def write_log(self, directory = "./logs", name = None):
        if not os.access(directory, os.F_OK):
            print(f"access to {directory} not allowed")
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        path = f'{directory}/{"log_" + time_str if name is None else name}.txt'

        self.object['n-epochs'] = self.epoch+1  # actual epochs = 0-indexed, this value = len() of that => 1-indexed 'count' of epochs
        string = json.dumps(self.object)
        with open(path, 'w') as file:
            file.write(string)

        print(f'output written to {path}')
        self.reset()
    '''

    def write_log(self):
        self.object['n-epochs'] = self.epoch+1  # actual epochs = 0-indexed, this value = len() of that => 1-indexed 'count' of epochs
        string = json.dumps(self.object)
        self.reset()
        return string
    
    '''
    @staticmethod
    def load_data(path):
        if not os.path.exists(path):
            print(f"{path} does not exist.")
            return
        with open(path, 'r') as f:
            data = json.loads(f.read())
        return data
    '''

    @staticmethod
    def load_data(string):
        data = json.loads(string)
        return data