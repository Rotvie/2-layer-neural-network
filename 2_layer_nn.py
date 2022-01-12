import numpy as np 
from numpy import load
import matplotlib.pyplot as plt

path = r'C:/Users/rotvi/Downloads/kagglecatsanddogs_3367a/PetImages/arrays/'

photos_train = load(path + 'dogs_vs_cats_photos_train.npy')
labels_train = load(path + 'dogs_vs_cats_labels_train.npy')
photos_test = load(path + 'dogs_vs_cats_photos_test.npy')
labels_test = load(path + 'dogs_vs_cats_labels_test.npy')

X_train = photos_train.reshape(photos_train.shape[0], -1).T
Y_train = np.reshape(labels_train, (1, labels_train.shape[0]))
X_test = photos_test.reshape(photos_test.shape[0], -1).T
Y_test = np.reshape(labels_test, (1, labels_test.shape[0]))

X_train = X_train/255.
X_test = X_test/255.
epsilon = 1e-6
print(X_train.shape, Y_train.shape) 
print(X_test.shape, Y_test.shape)  

n_x = X_train.shape[0]     # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

def sigmoid(Z):
    cache = Z
    A = 1/(1+np.exp(-Z))
    return A, cache

def relu(Z):
    cache = Z
    A = np.maximum(Z, 0.0)
    return A, cache

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters 

def linear_forward(A, W, b):
    Z = np.dot(W,A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    return Z, cache 

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b) # This "linear_cache" contains (A_prev, W, b)
        A, activation_cache = sigmoid(Z) # This "activation_cache" contains "Z"

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b) # This "linear_cache" contains (A_prev, W, b)
        A, activation_cache = relu(Z) # This "activation_cache" contains "Z"
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1/m) * (np.dot(Y, np.log(AL+epsilon).T) + np.dot((1-Y), np.log(1-AL+epsilon).T))
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    return cost

def relu_backward(dA, activation_cache):
    gprime = activation_cache
    gprime[gprime <= 0] = 0
    gprime[gprime > 0] = 1
    dZ = np.multiply(dA, gprime)
    return dZ

def sigmoid_backward(dA, activation_cache):
    gprime = activation_cache 
    A, cache = sigmoid(gprime)
    gprime = np.multiply(A,1-A)
    dZ = np.multiply(dA, gprime)
    return dZ

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters

def predict(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
    A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
    for i in range(A2.shape[1]):
        
        if A2[0,i] >= 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction


def two_layer_model(X, Y, layers_dims, learning_rate = 0.001, num_iterations = 500, print_cost=False):
    grads = {}
    costs = []                             
    m = X.shape[1]                          
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        cost = compute_cost(A2, Y)
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))   # Initializing backward propagation
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, learning_rate)
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    
    Y_prediction_train = predict(X_train, parameters)
    Y_prediction_test = predict(X_test, parameters)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
        
    return parameters

parameters = two_layer_model(X_train, Y_train, layers_dims = (n_x, n_h, n_y), num_iterations = 15, print_cost=True)


