import numpy as np
import matplotlib.pyplot as plt
from MNISTloader import MNISTloader
import random
from ActivationFunction import ActivationFunction as AF
from CostFunction import CostFunction as CF
array = np.array
random = np.random
exp = np.exp
dot = np.dot


class Network(object):

    def __init__(self, sizes):

        self.totalcost = []
        self.accuracy = []

        self.cost = CF.MSE()
        self.activ = AF.sigmoid()
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [random.randn(x, y)/np.sqrt(y) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        self.biases = [random.randn(x) for x in self.sizes[1:]]

    def feedforward(self, a, type="class"):
        self.a = []
        self.z = []
        # Classification uses only sigmoid activation
        if type == "class":

            for w, b in zip(self.weights, self.biases):
                z = dot(a, w) + b
                a = self.activ.sig(z)
                self.z.append(z)
                self.a.append(a)
            return a

        # Regression uses linear combination for the output
        # will have to change the gradient of the output error for this one
        elif type == "reg":

            for k in range(0, self.num_layers - 2):
                z = dot(a, self.weights[k]) + self.biases[k]
                a = self.activ.sig(z)
                self.z.append(z)
                self.a.append(a)
            z = (dot(a, self.weights[k + 1])
                 + self.biases[k + 1].flatten())
            self.z.append(z)
            self.a.append(z)
            return z

    def backpropogate(self, train_in, train_out):
        d = []
        wUpdate = []
        bUpdate = []

        # Get output error & initialize activity
        error = (self.cost).delta(self.feedforward(train_in), train_out, self.z[-1])
        active = [train_in] + self.a
        d.append(error)
        # backpropogation step
        for k in range(1, self.num_layers - 1):
            d.append(dot(self.weights[-k], error) * self.activ.sigprime(self.z[-(k + 1)]))
            error = d[k]
        d = list(reversed(d))

        # create weight updates
        for k in range(1, net.num_layers):
            wUpdate.append(dot(array([active[-(k + 1)]]).T, array([d[-k]])))
        wUpdate = list(reversed(wUpdate))

        # create bias updates
        for k in range(1, net.num_layers):
            bUpdate.append(d[-k])
        bUpdate = list(reversed(bUpdate))

        return wUpdate, bUpdate

    def update_batch(self, batch_in, batch_out, alpha):
        for x, y in zip(batch_in, batch_out):
            wUp, bUp = self.backpropogate(x, y)
            self.weights = [w - (alpha / len(batch_in)) * wp for w, wp in zip(self.weights, wUp)]
            self.biases = [b - (alpha / len(batch_in)) * bp for b, bp in zip(self.biases, bUp)]

    def SGD(self, train_X, train_Y, num_iter, batch_size, alpha):

        train = list(zip(train_X, train_Y))
        for k in range(0, num_iter):
            random.shuffle(train)
            train_X, train_Y = zip(*train)
            batch_X = [train_X[k:k + batch_size] for k in range(0, len(train_X), batch_size)]
            batch_Y = [train_Y[k:k + batch_size] for k in range(0, len(train_Y), batch_size)]

            # batch update loop
            for bX, bY in zip(batch_X, batch_Y):
                self.update_batch(bX, bY, alpha)

            # Output information about accuracy
            eva = self.evaluate(train_X, train_Y)
            self.accuracy.append(eva / len(train))
            print("Epoch: {0}: {1} / {2}".format(k+1,
                                                 eva,
                                                 len(train)))
            self.totalcost.append(CF.MSE.f(self.feedforward(train_X), train_Y))
        print("Accuracy is at {:0.2%}".format(eva / len(train)))

    def evaluate(self, train_X, train_Y):
        count = 0
        for k in range(0, len(train_X)):
            true = (np.argmax(train_Y[k]) + 1) % 10
            predict = (np.argmax(net.feedforward(train_X[k])) + 1) % 10
            if true == predict:
                count += 1
        return count


# Data import
X_train, y_train = MNISTloader.load_mnist(r"D:\Google Drive\Python\Neural Network\MNIST", kind='train')

# Initialization of network
net = Network([784, 100, 10])

# Preprocess Data for input
N = 1000
train_X = X_train[:N] / 255
train_Y = []
for k in range(0, len(y_train)):
    train_Y.append((np.zeros(10)))
    train_Y[k][y_train[k] - 1] = 1
train_Y = array(train_Y[:N])

# Run Stochastic Gradient Descent
net.SGD(train_X, train_Y, 50, 25, 0.5)

# Plot Results
plt.subplot(2,1,1)
plt.plot(range(0, 50), net.totalcost)
plt.xlabel("Epochs")
plt.ylabel("Cost")

plt.subplot(2, 1, 2)
plt.plot(range(0, 50), net.accuracy)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()