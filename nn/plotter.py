import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
from nn.trainer import Logger
import os, time

class Plotter:
    def read_file(self, string):
        self.data = Logger.load_data(string)
        if self.data is None:
            print("unsucessful.")
        else:
            print("JSON string read.")
    
    @staticmethod
    def skip_samples(array, limit):
        # can discard a few points from the end of the array
        has = array.shape[0]
        initial_skip = has // limit
        if initial_skip == 0:
            return array
        array = array[::initial_skip]

        first = array.shape[0] - 2 * (array.shape[0] - limit)
        if first == array.shape[0]:
            return array
        array = np.vstack((array[:first], array[first::2]))
        return array
    
    def plot_gradients(self, dir, name = None, n_points = None):
        if not os.access(dir, os.F_OK):
            print(f'cannot access {dir}')
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        name = "/gradients_" + (time_str if name is None else name) + ".png"

        weights_gradients = dict()
        bias_gradients = dict()

        for i in range(self.data['n-epochs']):
            epoch = f'epoch-{i}'
            for j in range(self.data[epoch]['n-updates']):
                update = f'update-{j}'
                for w in self.data['n_weights']:
                    weight = f'weights-gradient-{w}'
                    if weight not in weights_gradients:
                        weights_gradients[weight] = list()
                    array = np.asarray(self.data[epoch][update][weight]).ravel()[:4]
                    if array.shape[0] > 1:
                        array = array.reshape((2, -1))
                    else:
                        array = array.reshape((1, 1))
                    weights_gradients[weight].append(array)
                for b in self.data['n_weights']:
                    b += 1
                    bias = f'bias-gradient-{b}'
                    if bias not in bias_gradients:
                        bias_gradients[bias] = list()
                    array = np.asarray(self.data[epoch][update][bias]).ravel()[:4]
                    array = array.reshape((-1, 1))
                    bias_gradients[bias].append(array)
        
        fig, axs = plt.subplots(2, len(self.data['n_weights']), figsize = (7 * len(self.data['n_weights']), 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3}, squeeze = False)
        fig.suptitle(f'Gradients', size = 'xx-large')

        # weights
        for ax in range(len(self.data['n_weights'])):
            weight = f"weights-gradient-{self.data['n_weights'][ax]}"
            weights_gradients[weight] = np.asarray(weights_gradients[weight])  # convert entire history of gradients into one numpy array
            if n_points is not None:
                weights_gradients[weight] = weights_gradients[weight][:n_points]
            for r in range(weights_gradients[weight][0].shape[0]):
                for c in range(weights_gradients[weight][0].shape[1]):
                    axs[0, ax].plot(weights_gradients[weight][:, r, c], label = f'{r*2 + c}', linewidth = 0.7)
                    axs[0, ax].legend(title = 'elements')
                    axs[0, ax].set_title(f"weights-{self.data['n_weights'][ax]}")
                    axs[0, ax].set_xlabel("Updates")
        
        # bias
        for ax in range(len(self.data['n_weights'])):
            bias = f"bias-gradient-{self.data['n_weights'][ax] + 1}"
            bias_gradients[bias] = np.asarray(bias_gradients[bias])
            if n_points is not None:
                bias_gradients[bias] = bias_gradients[bias][:n_points]
            for r in range(bias_gradients[bias][0].shape[0]):
                    axs[1, ax].plot(bias_gradients[bias][:, r, 0], label = f'{r}', linewidth = 0.7)
                    axs[1, ax].legend(title = 'elements')
                    axs[1, ax].set_title(f"bias-{self.data['n_weights'][ax] + 1}")
                    axs[1, ax].set_xlabel("Updates")
        
        fig.savefig(dir + name, bbox_inches = "tight")
        print(f'plot saved at {dir + name}')
        plt.show()  # awesome pop-up window when using vs code!
    
    def plot_weights(self, dir, name = None, n_points = None):
        if not os.access(dir, os.F_OK):
            print(f'cannot access {dir}')
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        name = "/weights_" + (time_str if name is None else name) + ".png"

        weights = dict()
        bias = dict()

        init = self.data['init']
        for i in self.data['n_weights']:
            weight = f'weights-{i}'
            if weight not in weights:
                weights[weight] = list()
            array = np.asarray(init['weights'][weight]).ravel()[:4]
            if array.shape[0] > 1:
                array = array.reshape((2, -1))
            else:
                array = array.reshape((1, 1))
            weights[weight].append(array)
        for i in self.data['n_weights']:
            bias_name = f'bias-{i+1}'
            if bias_name not in bias:
                bias[bias_name] = list()
            array = np.asarray(init['bias'][bias_name]).ravel()[:4]
            array = array.reshape((-1, 1))
            bias[bias_name].append(array)

        for i in range(self.data['n-epochs']):
            epoch = f'epoch-{i}'
            for j in range(self.data[epoch]['n-updates']):
                update = f'update-{j}'
                for w in self.data['n_weights']:
                    weight = f'weights-{w}'
                    array = np.asarray(self.data[epoch][update][weight]).ravel()[:4]
                    if array.shape[0] > 1:
                        array = array.reshape((2, -1))
                    else:
                        array = array.reshape((1, 1))
                    weights[weight].append(array)
                for b in self.data['n_weights']:
                    bias_name = f'bias-{b+1}'
                    array = np.asarray(self.data[epoch][update][bias_name]).ravel()[:4]
                    array = array.reshape((-1, 1))
                    bias[bias_name].append(array)
        
        fig, axs = plt.subplots(2, len(self.data['n_weights']), figsize = (7 * len(self.data['n_weights']), 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3}, squeeze = False)
        fig.suptitle("Weights & Bias Values", size = 'xx-large')
        # weights
        for ax in range(len(self.data['n_weights'])):
            weight = f"weights-{self.data['n_weights'][ax]}"
            weights[weight] = np.asarray(weights[weight])
            if n_points is not None:
                weights[weight] = weights[weight][:n_points]
            for r in range(weights[weight][0].shape[0]):
                for c in range(weights[weight][0].shape[1]):
                    axs[0, ax].plot(weights[weight][:, r, c], label = f'{r*2 + c}', linewidth = 0.7)
                    axs[0, ax].legend(title = 'elements')
                    axs[0, ax].set_title(f"weights-{self.data['n_weights'][ax]}")
                    axs[0, ax].set_xlabel("Updates")
        
        # bias
        for ax in range(len(self.data['n_weights'])):
            bias_name = f"bias-{self.data['n_weights'][ax] + 1}"
            bias[bias_name] = np.asarray(bias[bias_name])
            if n_points is not None:
                bias[bias_name] = bias[bias_name][:n_points]
            for r in range(bias[bias_name][0].shape[0]):
                    axs[1, ax].plot(bias[bias_name][:, r, 0], label = f'{r}', linewidth = 0.7)
                    axs[1, ax].legend(title = 'elements')
                    axs[1, ax].set_title(f"bias-{self.data['n_weights'][ax] + 1}")
                    axs[1, ax].set_xlabel("Updates")
        
        fig.savefig(dir + name, bbox_inches = "tight")
        print(f'plot saved at {dir + name}')
        plt.show()
    
    def plot_score(self, dir, name = None, n_points = None, confusion_matrix = True):
        if not os.access(dir, os.F_OK):
            print(f'cannot access {dir}')
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        name = "/score_" + (time_str if name is None else name) + ".png"

        score_list = list()
        loss_list = list()

        for i in range(self.data['n-epochs']):
            epoch = f'epoch-{i}'
            score_list.append(self.data[epoch]['score'])
            loss_list.append(self.data[epoch]['loss'])

        if confusion_matrix:
            confusion_matrix = {'tp': list(), 'tn': list(), 'fp': list(), 'fn': list()}
            for i in range(self.data['n-epochs']):
                matrix = self.data[f'epoch-{i}']['confusion-matrix']
                for key in matrix.keys():
                    confusion_matrix[key].append(matrix[key])

        if confusion_matrix:
            fig, axs = plt.subplots(1, 2, figsize = (15, 5), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        else:
            fig, axs = plt.subplots(figsize = (8, 6))
            axs = np.array([axs])
        fig.suptitle(f"Metrics", size = 'xx-large')

        loss_list = np.asarray(loss_list)
        score_list = np.asarray(score_list)
        if n_points is not None:
                loss_list = loss_list[:n_points]
                score_list = score_list[:n_points]
        axs[0].plot(loss_list, label = "Loss")
        axs[0].plot(score_list, label = "Score")
        axs[0].set_xlabel("Epochs")
        axs[0].set_title("Loss & Score")
        axs[0].legend()

        if confusion_matrix:
            for key in confusion_matrix.keys():
                confusion_matrix[key] = np.asarray(confusion_matrix[key])
                if n_points is not None:
                    confusion_matrix[key] = confusion_matrix[key][:n_points]
                axs[1].plot(confusion_matrix[key], label = key)
            axs[1].set_xlabel("Epochs")
            axs[1].set_title("Confusion Matrix")
            axs[1].legend()

        fig.savefig(dir + name, bbox_inches = "tight")
        print(f'plot saved at {dir + name}')
        plt.show()
    
    @staticmethod
    def plot_points(x, y):
        fig, ax = plt.subplots(figsize = (7, 5))
        fig.suptitle(f"Points", size = 'xx-large')

        ax.scatter(x, y)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.show()
        
    def plot_predictions(self, features, dir, name = None, predictions = None):
        if not os.access(dir, os.F_OK):
            print(f'cannot access {dir}')
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        name = "/predictions_" + (time_str if name is None else name) + ".png"

        if predictions is not None:
            fig, axs = plt.subplots(figsize = (6, 4))
            fig.suptitle("Predictions", size = "xx-large")

            zeros = (predictions == 0)  # use this boolean array to index features
            ones = (predictions == 1)
            axis.scatter(features[zeros, 0], features[zeros, 1], c = "red", label = "0")
            axis.scatter(features[ones, 0], features[ones, 1], c = "blue", label = "1")
            axis.legend()

            fig.savefig(dir + name, bbox_inches = "tight")
            print(f'plot saved at {dir + name}')
            plt.show()
            return

        fig, axs = plt.subplots(2, 2, figsize = (12, 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle("Predictions", size = "xx-large")

        epochs = np.linspace(0, self.data['n-epochs']-1, 4, dtype = int)
        for i in range(epochs.shape[0]):
            axis = axs[i//2][i%2]
            epoch = f'epoch-{epochs[i]}'
            predictions = np.asarray(self.data[epoch]['predictions'], dtype = int)
            zeros = (predictions == 0)  # use this boolean array to index features
            ones = (predictions == 1)
            features = features[:predictions.shape[0]]
            axis.scatter(features[zeros, 0], features[zeros, 1], c = "red", label = "0")
            axis.scatter(features[ones, 0], features[ones, 1], c = "blue", label = "1")
            axis.set_title(f'Epoch-{epochs[i]}')
            axis.legend()

        fig.savefig(dir + name, bbox_inches = "tight")
        print(f'plot saved at {dir + name}')
        plt.show()
    
    def plot_classification(self, x_axis, y_axis, targets, dir, name = None):
        if not os.access(dir, os.F_OK):
            print(f'cannot access {dir}')
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        name = "/classification_" + (time_str if name is None else name) + ".png"

        fig, axs = plt.subplots(2, 2, figsize = (12, 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle("Classification: First 200 samples", size = "x-large")


        epochs = np.linspace(0, self.data['n-epochs']-1, 4, dtype = int)
        for i in range(epochs.shape[0]):
            axis = axs[i//2][i%2]
            epoch = f'epoch-{epochs[i]}'

            predictions = np.asarray(self.data[epoch]['predictions'], dtype = int)
            shorter = min(predictions.shape[0], x_axis.shape[0])
            predictions = predictions[:shorter]
            x_axis = x_axis[:shorter]
            y_axis = y_axis[:shorter]
            targets = targets[:shorter]
            targets = targets.reshape(predictions.shape)

            zeros = (predictions == 0)  # use this boolean array to index features
            ones = (predictions == 1)
            incorrect = ((targets == 1) & zeros) | ((targets == 0) & ones)

            axis.scatter(x_axis[zeros], y_axis[zeros], c = "red", label = "0")
            axis.scatter(x_axis[ones], y_axis[ones], c = "blue", label = "1")
            axis.scatter(x_axis[incorrect], y_axis[incorrect], c = "black", label = "incorrect", marker = "x")
            axis.set_xlabel("PCA Component-1")
            axis.set_ylabel("PCA Component-2")
            axis.set_title(f'Epoch-{epochs[i]}')
            axis.legend()

        fig.savefig(dir + name, bbox_inches = "tight")
        print(f'plot saved at {dir + name}')
        plt.show()

    def plot_regression(self, features, targets, dir, name = None, predictions = None, x_label = None):
        if not os.access(dir, os.F_OK):
            print(f'cannot access {dir}')
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        name = "/regression_" + (time_str if name is None else name) + ".png"

        features = features.reshape((features.shape[0], 1))
        targets = targets.reshape((targets.shape[0], 1))
        prediction_size = 15

        if predictions is not None:
            fig, ax = plt.subplots(figsize = (6, 4))
            fig.suptitle("Predictions", size = "xx-large")

            ax.scatter(features[:, 0], targets[:, 0], label = "targets", c = "cornflowerblue")
            ax.scatter(features[:, 0], predictions, label = "predictions", marker = "x", c = "coral", s = prediction_size)
            ax.set_xlabel(x_label)
            ax.legend()

            fig.savefig(dir + name, bbox_inches = "tight")
            print(f'plot saved at {dir + name}')
            plt.show()
            return

        fig, axs = plt.subplots(2, 2, figsize = (12, 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle("Predictions", size = "xx-large")

        epochs = np.linspace(0, self.data['n-epochs']-1, 4, dtype = int)

        for i in range(epochs.shape[0]):
            axis = axs[i//2][i%2]
            epoch = f'epoch-{epochs[i]}'
            predictions = np.asarray(self.data[epoch]['predictions'])
            features = features[:predictions.shape[0], :]
            targets = targets[:predictions.shape[0], :]
            axis.scatter(features[:, 0], targets[:, 0], label = "targets", c = "cornflowerblue")
            axis.scatter(features[:, 0], predictions, label = "predictions", marker = "x", c = "coral", s = prediction_size)
            axis.set_title(f'Epoch-{epochs[i]}')
            axis.set_xlabel(x_label)
            axis.legend()
        
        fig.savefig(dir + name, bbox_inches = "tight")
        print(f'plot saved at {dir + name}')
        plt.show()

    def plot_contours(self, trainer, features, targets, dir, name = None, magnitude = 1, seed = 91):
        if not os.access(dir, os.F_OK):
            print(f'cannot access {dir}')
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        name = "/contours_" + (time_str if name is None else name) + ".png"

        model = trainer.model
        original_vector = model.weights_elements()

        seeds = default_rng(seed).choice(1000, (4, 2), replace = False)
        fig, axs = plt.subplots(2, 2, figsize = (12, 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle("Loss Landscape", size = "xx-large")

        x = np.linspace(-magnitude, magnitude, 20)
        y = np.linspace(-magnitude, magnitude, 20)
        z = np.zeros((y.size, x.size))  # as per matplotlib's contour plot signature

        features = features[:200, :]
        targets = targets[:200, :]

        for ax in range(4):
            vector1 = default_rng(seeds[ax][0]).uniform(-1, 1, size = original_vector.shape)
            vector1 = (vector1 / np.linalg.norm(vector1)) * np.linalg.norm(original_vector)
            vector2 = default_rng(seeds[ax][1]).uniform(-1, 1, size = original_vector.shape)
            vector2 = (vector2 / np.linalg.norm(vector2)) * np.linalg.norm(original_vector)

            for i in range(y.size):  # y first
                for j in range(x.size):
                    model.set_weights(original_vector + y[i] * vector1 + x[j] * vector2)
                    _, loss, _ = trainer.predict(features, targets, quiet = True)
                    z[i, j] = loss
            
            axis = axs[ax//2][ax%2]
            # levels = 15 (earlier) --> np.linspace(np.min(z), np.max(z)//4, 10)
            contour_set = axis.contour(x, y, z, levels = 15, cmap = "magma")
            axis.clabel(contour_set)
            axis.set_xlabel("alpha")
            axis.set_ylabel("beta")
            axis.scatter(0, 0, c = "blue", label = "model")
            axis.legend()
            print(f'Grid-{ax} done')
        
        model.set_weights(original_vector)
        fig.savefig(dir + name, bbox_inches = "tight")
        print(f'plot saved at {dir + name}')
        plt.show()