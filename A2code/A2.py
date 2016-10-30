""" ENGG*6500 A2
    Train a MLP on the StumbleUpon Evergreen dataset
"""

import time
import numpy as np
import pickle
import matplotlib.pyplot as plt

from numpy.random import uniform
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import multigrad


# Global variables
dtype = 'float32'
eps = np.finfo(np.double).eps  # -- a small number


def load_evergreen(dtype=dtype):
    with open('numerical_features.pkl') as f:
        train_set, val_set, test_set = pickle.load(f)
    train_X, train_y = train_set
    val_X, val_y = val_set
    # test set has ids instead of lables
    test_X, test_ids = test_set

    # append the lda topic features
    with open('lda_features.pkl') as f:
        train_w, val_w, test_w = pickle.load(f)
    train_X = np.concatenate((train_X, train_w), axis=1)
    val_X   = np.concatenate((val_X, val_w), axis=1)
    test_X  = np.concatenate((test_X, test_w), axis=1)

    return (train_X.astype(dtype), train_y.astype('int8'),
            val_X.astype(dtype), val_y.astype('int8'),
            test_X.astype(dtype), test_ids)


def logistic(z):
    """return logistic sigmoid of float or ndarray `z`"""
    return 1.0 / (1.0 + np.exp(-z))


def p_y_given_x(W, b, x):
    return logistic(np.dot(x, W) + b)


def logreg_prediction(W, b, x):
    return p_y_given_x(W, b, x) > 0.5


def cross_entropy(x, z):
    # note we add a small epsilon for numerical stability
    return -(x * np.log(z + eps) + (1 - x) * np.log(1-z + eps))


def logreg_cost(W, b, x, y):
    z = p_y_given_x(W, b, x)
    l = cross_entropy(y, z).mean(axis=0)
    return l


def accuracy(y, y_pred):
    return 1.0*np.sum(y == y_pred)/y.shape[0]


def mlp_cost(X, y, W_hid, b_hid, W_out, b_out):
    # forward pass
    # hidden activations
    act_hid = p_y_given_x(W_hid, b_hid, X)
    # output activation
    act_out = p_y_given_x(W_out, b_out, act_hid)
    return cross_entropy(y, act_out).mean(axis=0)


def mlp_predict(X, W_hid, b_hid, W_out, b_out):
    act_hid = p_y_given_x(W_hid, b_hid, X)
    act_out = p_y_given_x(W_out, b_out, act_hid)
    return act_out > 0.5


def initialize_model(n_inputs, n_hidden, dtype=dtype):
    W_hid = uniform(low=-4*np.sqrt(6.0 / (n_inputs + n_hidden)),
                    high=4*np.sqrt(6.0 / (n_inputs + n_hidden)),
                    size=(n_inputs, n_hidden)).astype(dtype)
    b_hid = np.zeros(n_hidden, dtype=dtype)

    # now allocate the logistic regression model at the top
    W_out = uniform(low=-4*np.sqrt(6.0 / (n_inputs+n_hidden)),
                    high=4*np.sqrt(6.0 / (n_inputs+n_hidden)),
                    size=(n_hidden,)).astype(dtype)
    b_out = np.array(0.0)

    return W_hid, b_hid, W_out, b_out


def train_model(train_X, train_y, val_X, val_y, W_hid, b_hid, W_out, b_out,
                learning_rate, epochs, batch_size=100, dtype=dtype):
    train_costs = np.zeros(epochs, dtype=dtype)
    val_costs   = np.zeros(epochs, dtype=dtype)

    # Make a list of the weights
    weights = [W_hid, b_hid, W_out, b_out]

    num_batches = train_X.shape[0] / batch_size

    mlp_grads = multigrad(mlp_cost, [2, 3, 4, 5])

    for epoch in xrange(epochs):
        print "Epoch", epoch

        for bi in xrange(num_batches):
            grads = mlp_grads(train_X[batch_size*bi:batch_size*(bi+1),:],
                              train_y[batch_size*bi:batch_size*(bi+1)],
                              *weights)

            # returns a CudaNDarray when running on GPU
            # creating a np.array copies the data back from the GPU
            if not isinstance(grads[0], np.ndarray):
                grads = [np.array(g) for g in grads]

            # update the weights
            for i in xrange(len(weights)):
                weights[i] -= learning_rate * grads[i]

        train_cost = mlp_cost(train_X, train_y, *weights)
        val_cost = mlp_cost(val_X, val_y, *weights)
        train_costs[epoch] = train_cost
        val_costs[epoch]   = val_cost

        # print "Training set cost:  ", train_cost
        # print "Validation set cost:", val_cost

    return train_costs, val_costs


def run_training(n_hidden, learning_rate, epochs, data=None, model=None,
                 batch_size=100, show_plot=True):
    t0 = time.time()
    train_X, train_y, val_X, val_y = data

    # initialize input layer parameters
    n_inputs = train_X.shape[1]  # -- aka D_0
    print "NUM input dimensions:", n_inputs

    if model is None:
        model = initialize_model(n_inputs, n_hidden)
    W_hid, b_hid, W_out, b_out = model
    #
    # print "Before training"
    # print 'train accuracy: %6.4f' % \
    #       accuracy(train_y, mlp_predict(train_X, W_hid, b_hid, W_out, b_out))
    # print 'train cross entropy: %6.4f' % \
    #       mlp_cost(train_X, train_y, W_hid, b_hid, W_out, b_out)
    # print 'validation accuracy: %6.4f' % \
    #       accuracy(val_y, mlp_predict(val_X, W_hid, b_hid, W_out, b_out))
    # print 'validation cross entropy: %6.4f' % mlp_cost(val_X, val_y, W_hid,
    #                                                    b_hid, W_out, b_out)

    train_costs, val_costs = train_model(train_X, train_y, val_X, val_y,
                                         W_hid, b_hid, W_out, b_out,
                                         learning_rate, epochs, batch_size)

    print "After training, n_hidden: %s, learning_rate: %s" % (n_hidden,
                                                               learning_rate)
    print 'train accuracy: %6.4f' % \
          accuracy(train_y, mlp_predict(train_X, W_hid, b_hid, W_out, b_out))
    print 'train cross entropy: %6.4f' % \
          mlp_cost(train_X, train_y, W_hid, b_hid, W_out, b_out)
    print 'validation accuracy: %6.4f' % \
          accuracy(val_y, mlp_predict(val_X, W_hid, b_hid, W_out, b_out))
    print 'validation cross entropy: %6.4f' % mlp_cost(val_X, val_y, W_hid,
                                                       b_hid, W_out, b_out)

    print "training took: %s sec" % (time.time() - t0)

    if show_plot:
        plt.plot(train_costs, '-b', label="Training data")
        plt.plot(val_costs, '-r', label="Validation data")
        plt.legend(loc='upper right')
        plt.xlabel("Epoch")
        plt.ylabel("Cross entropy")
        plt.show()

    return train_costs, val_costs


def write_predictions(model, test_X, test_ids, outfile):
    '''Save the predcition to a file.'''
    y_pred = mlp_predict(test_X, *model)

    with open(outfile, 'wb') as f:
        f.write('urlid,label\n')
        for i in xrange(len(test_ids)):
            label = int(y_pred[i])
            urlid = test_ids[i]
            f.write('%s,%s\n' % (urlid, label))


if __name__ == '__main__':

    epochs = 150
    learning_rate = 0.01
    n_hidden = 100
    batch_size = 500

    data = load_evergreen()
    # load_evergreen returns a tuple:
    # (train_X, train_y, val_X, val_y, test_X, test_ids)
    # slice to seperate the test data
    test_X, test_ids = data[-2:]
    data   = data[:-2]

    n_inputs = data[0].shape[1]
    model = initialize_model(n_inputs, n_hidden)

    train_costs, val_costs = run_training(n_hidden, learning_rate, epochs,
                                          data, model, batch_size,
                                          show_plot=False)

    # write_predictions(model, test_X, test_ids,
    #                  'pred_nhid_%s_lr_%s_epochs_%s.txt' % (n_hidden,
    #                                                        learning_rate,
    #                                                        epochs))
