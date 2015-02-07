"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from mlp_seg_simple33 import HiddenLayer
import h5py
from logistic_sgd_seg_print_h5 import LogisticRegression,load_data
from loadh5_logistic import load_weight


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), params_file=''):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        #num.prod: production compute
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        if params_file:
            params = load_weight(params_file)
            weight = params[0]
            bias = params[1]

            self.W = theano.shared(
                value=numpy.asarray(weight,dtype=theano.config.floatX),
                name='W',
                borrow=True
            )
            # initialize the baises b as a vector of n_out 0s
            self.b = theano.shared(
                value=numpy.asarray(bias, dtype=theano.config.floatX),
                name='b',
                borrow=True
            )
        else:
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

            # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]
    def parameters(self):
        return self.params


def evaluate_lenet5(learning_rate=0.1, n_epochs=2,
                    dataset='mnist.pkl.gz',
                    nkerns=[16, 16], batch_size=512,
                    params_file0='layer0.h5', params_file1='layer1.h5', params_file2='layer2.h5',
                    params_file3='layer3.h5'):
                    #params_file0='', params_file1='', params_file2='', params_file3=''):
                    #nkerns=[20, 50], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    #load segmentation data
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] #the 1st dim, means #of item
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    print "n_train_batches, valid, test", n_train_batches, n_valid_batches,n_test_batches 
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    #(95,95) is the size of pixel and its neighbor
    #(65,65) is the size of pixel and its neighbor
    layer0_input = x.reshape((batch_size, 1, 33, 33))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)

    # filtering reduces the image size to (65-4+1 , 65-4+1) = (62, 62)
    # maxpooling reduces this further to (62/2, 62/2) = (31, 31)
    #Todo I will automatic compute this
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 33, 33),
        #image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 4, 4),
        #filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2),
        params_file = params_file0
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 4, 4)

    # filtering reduces the image size to (31-4+1, 31-4+1) = (28, 28)
    # maxpooling reduces this further to (28/2, 28/2) = (14,14)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 15, 15),
        #image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 4, 4),
        poolsize=(2, 2),
        params_file = params_file1
    )

    # Construct the third convolutional pooling layer
    # filtering reduces the image size to (14-5+1, 14-5+1) = (10, 10)
    # maxpooling reduces this further to (10/2, 10/2) = (5, 5)
    # 4D output tensor is thus of shape (nkerns[0], nkerns[1], 5, 5)


    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.

    # This will generate a matrix of shape (batch_size, nkerns[1] * 3 * 3),
    # or (50, 48 * 3 * 3) = (50, 48*9) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 6 * 6, #!! n_in is number of inputs
        #n_in=nkerns[1] * 4 * 4,
        n_out=100,
        #n_out=500,
        activation=T.tanh,
        params_file = params_file2
    )

    # classify the values of the fully-connected sigmoidal layer
    #layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    layer3 = LogisticRegression(input=layer2.output, n_in=100, n_out=2, params_file=params_file3)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)
    #cost = layer3.negative_log_likelihood(y)
    print "build a last layer, log likelyhood......"
    # create a function to compute the mistakes that are made by the model
    print type(test_set_x), type(test_set_y)
    test_model = theano.function(
        [index],
        layer3.errors(y),
        #layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    predict_model = theano.function(
        [index],
        layer3.predict(y),
        #layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        #layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    #patience = 100  # look as this many examples regardless
    patience = 1000  # look as this many examples regardless
    #patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0 :
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    print "testing....."
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    #mypredict0 = [predict_model(i) for i in xrange(n_test_batches)]
    f = open("cnn33_predict.txt",'w')
    for i in range(n_test_batches):
      mypredict0 = predict_model(i)
      mypredict = mypredict0[0]
      for ele in mypredict:
        f.write(str(ele))
        f.write(" ")
      f.write("\n")
    f.close()
    #print predict_y
    params0 = layer0.parameters()
    params1 = layer1.parameters()
    params2 = layer2.parameters()
    params3 = layer3.parameters()

    layer0_weight =  params0[0].get_value(borrow=True)
    layer0_bias =  params0[1].get_value(borrow=True)

    layer1_weight =  params1[0].get_value(borrow=True)
    layer1_bias =  params1[1].get_value(borrow=True)

    layer2_weight =  params2[0].get_value(borrow=True)
    layer2_bias =  params2[1].get_value(borrow=True)

    layer3_weight =  params3[0].get_value(borrow=True)
    layer3_bias =  params3[1].get_value(borrow=True)

    h5file = h5py.File("layer0.h5",'w')
    h5file['weight'] = layer0_weight
    h5file['bias'] = layer0_bias
    h5file.close()

    h5file = h5py.File("layer1.h5",'w')
    h5file['weight'] = layer1_weight
    h5file['bias'] = layer1_bias
    h5file.close()

    h5file = h5py.File("layer2.h5",'w')
    h5file['weight'] = layer2_weight
    h5file['bias'] = layer2_bias
    h5file.close()

    h5file = h5py.File("layer3.h5",'w')
    h5file['weight'] = layer3_weight
    h5file['bias'] = layer3_bias
    h5file.close()


if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
