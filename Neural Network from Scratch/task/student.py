import numpy as np
import pandas as pd
import os
import requests
import math
from matplotlib import pyplot as plt




def one_hot(data: np.ndarray) -> np.ndarray:
    y_train = np.zeros((data.size, data.max() + 1))
    rows = np.arange(data.size)
    y_train[rows, data] = 1
    return y_train

# Rescaled X_train, X_test: X_new = X/max(X)

def scale(X_train, X_test):
    mx = max(np.max(X_train), np.max(X_test))
    newX_train =  X_train/mx
    newX_test = X_test/mx
    return newX_train, newX_test

# Xavier initialization function
def xavier(n_in,  n_out):
    low = -np.sqrt(6)/np.sqrt(n_in+n_out)
    high = np.sqrt(6)/np.sqrt(n_in+n_out)
    w = np.random.uniform(low, high, (n_in, n_out))
    return w

# activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# derivative of activation function
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

# loss function MSE
def mse(y_pred,y_true):
    return np.mean(np.power(y_true-y_pred, 2));

# derivative of loss function MSE
def mse_prime(y_pred,y_true):
    return 2*(y_pred-y_true)

# Calculate accuracy
def accuracy(estimator, X, y): # estimator is model
    y_pred = np.argmax(estimator.forward(X), axis=1)
    y_true = np.argmax(y, axis=1)
    return np.mean(y_pred == y_true) # count all correct pred and divide by len(y_true)

# Perform a single epoch of training
def train(estimator, X, y, alpha, batch_size=100):
    n = X.shape[0]
    for i in range(0, n, batch_size):
        estimator.backprop(X[i:i + batch_size], y[i:i + batch_size], alpha)

class OneLayerNeural:

    def __init__(self, n_features, n_classes):
        # Initiate weights and biases using Xavier
        self.weights = xavier(n_features, n_classes)
        self.bias = xavier(1, n_classes)

    def forward(self, X):
        # Perform a forward step
        return sigmoid(np.dot(X, self.weights) + self.bias)

    def backprop(self, X, y, alpha):
        # Perform a backward step
        # Calculate the error
        error = (mse_prime(self.forward(X), y) * sigmoid_prime(np.dot(X, self.weights) + self.bias))

        # Calculate the gradient
        delta_W = (np.dot(X.T, error)) / X.shape[0]
        delta_b = np.mean(error, axis=0)

        # Update weights and biases
        self.weights -= alpha * delta_W
        self.bias -= alpha * delta_b

class TwoLayerNeural():
    def __init__(self, n_features, n_classes):
        # Size of the hidden layer (64 neurons)
        hidden_size = 64
        # Initializing weights
        self.weights = [xavier(n_features, hidden_size), xavier(hidden_size, n_classes)]
        self.bias = [xavier(1, hidden_size), xavier(1, n_classes)]

    def forward(self, X):
        # Calculating feedforward
        z = X
        for i in range(2):
            z = sigmoid(np.dot(z, self.weights[i]) + self.bias[i])
        return z

    def backprop(self, X, y, alpha):
        # Number of trained samples
        n = X.shape[0]
        # Vector of ones for bias calculation
        biases = np.ones((1, n))
        # Result of output layer after forwarding
        yp = self.forward(X)

        # Calculate the gradient of the loss function with respect to the bias of the output layer
        loss_grad_1 = 2 * alpha / n * ((yp - y) * yp * (1 - yp))

        # Calculate the output of the first layer
        f1_out = sigmoid(np.dot(X, self.weights[0]) + self.bias[0])

        # Calculate the gradient of the loss function with respect to the bias of the first layer
        loss_grad_0 = np.dot(loss_grad_1, self.weights[1].T) * f1_out * (1 - f1_out)

        # Update weights and biases
        self.weights[0] -= np.dot(X.T, loss_grad_0)
        self.weights[1] -= np.dot(f1_out.T, loss_grad_1)

        self.bias[0] -= np.dot(biases, loss_grad_0)
        self.bias[1] -= np.dot(biases, loss_grad_1)

def plot(loss_history: list, accuracy_history: list, filename='plot'):

    # function to visualize learning process at stage 4

    n_epochs = len(loss_history)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Loss on train dataframe from epoch')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history)

    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0, n_epochs, 4))
    plt.title('Accuracy on test dataframe from epoch')
    plt.grid()

    plt.savefig(f'{filename}.png')


if __name__ == '__main__':

    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if ('fashion-mnist_train.csv' not in os.listdir('../Data') and
            'fashion-mnist_test.csv' not in os.listdir('../Data')):
        print('Train dataset loading.')
        url = "https://www.dropbox.com/s/5vg67ndkth17mvc/fashion-mnist_train.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_train.csv', 'wb').write(r.content)
        print('Loaded.')

        print('Test dataset loading.')
        url = "https://www.dropbox.com/s/9bj5a14unl5os6a/fashion-mnist_test.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/fashion-mnist_test.csv', 'wb').write(r.content)
        print('Loaded.')

    # Read train, test data.
    raw_train = pd.read_csv('../Data/fashion-mnist_train.csv')
    raw_test = pd.read_csv('../Data/fashion-mnist_test.csv')

    X_train = raw_train[raw_train.columns[1:]].values
    X_test = raw_test[raw_test.columns[1:]].values

    y_train = one_hot(raw_train['label'].values)
    y_test = one_hot(raw_test['label'].values)

    scaled_X_train, scaled_X_test = scale(X_train, X_test)

    # Create an instance of model
    #estimator = OneLayerNeural(scaled_X_train.shape[1], 10)
    # Create a class instance and train it
    estimator = TwoLayerNeural(scaled_X_train.shape[1], y_train.shape[1])
    # Perform backprop
    #estimator.backprop(scaled_X_train[:2], y_train[:2], 0.1)
    # Perform a forward step
    #y_pred = estimator.forward(scaled_X_train[:2])
    # Calculate the MSE
    #r1 = mse(y_pred, y_train[:2]).flatten().tolist()
    # Train the model
    #r1 = estimator.forward(scaled_X_train[:2]).flatten().tolist()
    # Test the accuracy
    #r1 = accuracy(estimator, scaled_X_test, y_test).flatten().tolist()
    # Train the model fo 20 epochs and add accuracies to a list
    r2 = []
    for _ in range(20):
        train(estimator, scaled_X_train, y_train, 0.5)
        r2.append(accuracy(estimator, scaled_X_test, y_test))
    # Perform a backward step (train the model)
    #model.backprop(scaled_X_train[:2], y_train[:2], 0.1)
    # Implement the forward step
    #y_pred = model.forward(scaled_X_train[:2])
    # Use the [−1,0,1,2] and [4,3,2,1] arrays to test your MSE and the MSE derivative functions
   # a1 = np.array([-1, 0, 1, 2])
    #a2 = np.array([4, 3, 2, 1])
    # Calculate MSE  and the derivative of MSE
    #r1 = mse(a1, a2).flatten().tolist()
    #r2 = mse_prime(a1, a2).flatten().tolist()
    #r3 = sigmoid_prime(a1).flatten().tolist()
    #r4 = mse(y_pred, y_train[:2]).flatten().tolist()


    print(r2)