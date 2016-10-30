""" ENGG*6500 A2
    Train a MLP on the StumbleUpon Evergreen dataset
"""

import time
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from numpy.random import uniform

### GLOBAL ###
DTYPE = 'float32'
EPS = np.finfo(np.double).eps  # -- a small number

### HyperParameters ###
flags = tf.flags
flags.DEFINE_string('raw_data', 'ratings.txt', 'path to raw data file')
flags.DEFINE_string('data', 'data.p', 'path to data file')
flags.DEFINE_string('model', 'model', 'path to model file')
flags.DEFINE_integer('n_hidden', 90, 'hidden layer size')
flags.DEFINE_integer('batch_size', 500, 'mini batch size')
flags.DEFINE_integer('max_epoch', 200, 'maximum number of epoch')
flags.DEFINE_float('lr', 0.33, 'learning rate')
FLAGS = flags.FLAGS


### MLP MODEL ###
class MLP(object):
    
    def __init__(self, sess, data):
        self.lr = FLAGS.lr
        self.epochs = FLAGS.max_epoch
        self.n_hidden = FLAGS.n_hidden
        self.n_inputs = data['trX'].shape[1] # aka D_0
        self.n_outputs = data['trY'].shape[1]

        params = self.def_param(self.n_inputs, self.n_hidden, self.n_outputs)

        self.W_hid = params['W_h']
        self.b_hid = params['b_h']
        self.W_out = params['W_o']
        self.b_out = params['b_o']
        self.x = params['in_x']
        self.y = params['lbl_y']
        
        
        ## Model Architecture ##
        h_in = tf.nn.bias_add(tf.matmul(self.x,self.W_hid),self.b_hid)
        # apply the non-linearity (Activation)
        h_out = tf.sigmoid(h_in)

        o_in = tf.nn.bias_add(tf.matmul(h_out, self.W_out), self.b_out)
         
        y_pred = tf.sigmoid(o_in)

        # cost = tf.nn.sigmoid_cross_entropy_with_logits(o_in, y)
        self.cost = -tf.reduce_mean(self.y * tf.log(y_pred + EPS) + \
                                    (1 - self.y) * tf.log(1 - y_pred + EPS))
        ## End Model Architecture ## 

        # Optimization Method
        self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.cost)
        self.accuracy, self.bin_pred = self.get_acc(sess, y_pred)
        

    def get_acc(self, sess, y_pred):
        # if element > 0.5 replace with 1.
        bool_vec = tf.greater(y_pred, 0.5)
        bin_y_pred = tf.cast(bool_vec, tf.float32)
        correct_prediction = tf.equal(self.y, bin_y_pred)
        # convert all boolean to float, to calculate percentage
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return accuracy, bin_y_pred


    def def_param(self, n_inputs, n_hidden, n_outputs):
        # initialization from Glorot and Bengio 2010.
        W_hid = tf.Variable(uniform(low=-4 * np.sqrt(6.0 / (n_inputs + n_hidden)), \
                high=4 * np.sqrt(6.0 / (n_inputs + n_hidden)),\
                size=(n_inputs,n_hidden)).astype(DTYPE), name='W_h')
        b_hid = tf.Variable(np.zeros([n_hidden], dtype=DTYPE), name='b_h')

        W_out = tf.Variable(uniform(low=-4 * np.sqrt(6.0 / (n_hidden + n_outputs)),\
                high=4 * np.sqrt(6.0 / (n_hidden + n_outputs)),\
                size=(n_hidden, 1)).astype(DTYPE), name='W_h')
        b_out =  tf.Variable(tf.zeros([1]), name='b_o')
        input_x = tf.placeholder("float", [None, n_inputs]) # create symbolic variables
        #The label.
        label_y = tf.placeholder("float", [None, 1])
        data_size = tf.placeholder("float", [None])
        params = {
            'W_o' : W_out,
            'b_o' : b_out,
            'W_h' : W_hid,
            'b_h' : b_hid,
            'in_x'   : input_x,
            'lbl_y'   : label_y,
            'data_size' : data_size
            }

        return params

    

def load_evergreen(dtype=DTYPE):
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
    train_y = train_y.reshape(len(train_y), 1)
    val_y = val_y.reshape(len(val_y), 1)

    out = {
        'trX' : train_X.astype(DTYPE),
        'trY' : train_y.astype(DTYPE),
        'vlX' : val_X.astype(DTYPE),
        'vlY' : val_y.astype(DTYPE),
        'tstX' : test_X.astype(DTYPE),
        'tstID' : test_ids
    }

    return out


def train(sess, model, data, show_plot):
    epochs = FLAGS.max_epoch
    batch_size = FLAGS.batch_size
    train_costs = np.zeros(epochs, dtype=DTYPE)
    test_costs = np.zeros(epochs, dtype=DTYPE)
    x = model.x
    y = model.y
    num_batches = data['trX'].shape[0] / batch_size

    t0 = time.time()
    print 'train accuracy before training: %6.4f' % \
            sess.run(model.accuracy, feed_dict={x: data['trX'], y: data['trY']})
    print 'train cross entropy before training: %6.4f'%  \
            sess.run(model.cost, feed_dict={x: data['trX'], y: data['trY']})
    print 'test accuracy before training: %6.4f' % \
            sess.run(model.accuracy, feed_dict={x: data['vlX'], y: data['vlY']})
    print 'test cross entropy before training: %6.4f' % \
            sess.run(model.cost, feed_dict={x: data['vlX'], y: data['vlY']})

    for i in xrange(epochs):
        for start, end in zip(range(0, len(data['trX']), batch_size), range(batch_size, len(data['trX']),batch_size)):
            sess.run(model.train_op, feed_dict={x: data['trX'][start:end], y: data['trY'][start:end]})
        
        train_cost = sess.run(model.cost,  feed_dict={x: data['trX'], y: data['trY']})
        test_cost = sess.run(model.cost, feed_dict={x: data['vlX'], y: data['vlY']})

        train_costs[i] = train_cost
        test_costs[i] = test_cost

        if i % 50 == 0:
            print "< Epoch%d >" % i
            print "Cross Entropy Cost: %6.4f" % (sess.run(model.cost, feed_dict={x: data['trX'], y: data['trY']}))
            print "Accuracy: ", sess.run(model.accuracy, feed_dict={x: data['vlX'], y: data['vlY']})
        
    print 'train accuracy after training: %6.4f' % \
            sess.run(model.accuracy, feed_dict={x: data['trX'], y: data['trY']})
    print 'train cross entropy after training: %6.4f'%  \
            sess.run(model.cost, feed_dict={x: data['trX'], y: data['trY']})
    print 'test accuracy after training: %6.4f' % \
            sess.run(model.accuracy, feed_dict={x: data['vlX'], y: data['vlY']})
    print 'test cross entropy after training: %6.4f'%  \
            sess.run(model.cost, feed_dict={x: data['vlX'], y: data['vlY']})
    print "training took: %s sec" % (time.time() - t0)

    if show_plot:
        plt.plot(train_costs, '-b', label="Training data")
        plt.plot(test_costs, '-r', label="Test data")
        plt.legend(loc='upper right')
        plt.xlabel("Epoch")
        plt.ylabel("Cross entropy cost")
        plt.show()


    return train_costs, test_costs


def write_predictions(sess,model, test_X, test_ids, outfile):
    pred = sess.run(model.bin_pred, feed_dict={model.x: test_X})

    with open(outfile, 'wb') as f:
        f.write('urlid,label\n')
        for i in xrange(len(test_ids)):
            label = int(pred[i])
            urlid = test_ids[i]
            f.write('%s,%s\n' % (urlid, label))



if __name__ == '__main__':

    data = load_evergreen()
    test_X = data['tstX']
    test_ids = data['tstID']

    sess = tf.Session()

    model = MLP(sess, data)

    sess.run(tf.initialize_all_variables())

    train(sess, model, data, show_plot = False)

    write_predictions(sess, model, test_X, test_ids,
                     'pred_nhid_%s_lr_%s_epochs_%s.txt' % (model.n_hidden,
                                                           model.lr,
                                                           model.epochs))
    sess.close()

