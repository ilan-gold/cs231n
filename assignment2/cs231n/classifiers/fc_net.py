from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        fc_cache, relu_cache = cache1[0], cache1[1]
        out2, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])
        scores = out2

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores
        grads = {}
        loss, softmax_grad = softmax_loss(scores, y)
        loss =  loss + self.reg * .5 * (np.sum(cache2[1] * cache2[1]) + np.sum(fc_cache[1] * fc_cache[1]))
        dx, grads['W2'], grads['b2'] = affine_backward(softmax_grad, cache2)
        _, grads['W1'], grads['b1'] = affine_relu_backward(dx, cache1)
        grads['W2'] = grads['W2'] + self.reg * .5 * 2 * cache2[1]
        grads['W1'] = grads['W1'] + self.reg * .5 * 2 * fc_cache[1]

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        layers = [input_dim] + hidden_dims + [num_classes]
        count = 1
        while count < len(layers):
            self.params['b' + str(count)] = np.zeros(layers[count])
            self.params['W' + str(count)] = weight_scale * np.random.randn(layers[count - 1], layers[count])
            if count != len(layers) - 1 and normalization == 'batchnorm':
                self.params['gamma' + str(count)] = np.ones(layers[count])
                self.params['beta' + str(count)] = np.zeros(layers[count])
            count = count + 1
        

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        out = X
        fc_cache, relu_cache, bn_cache, dropout_cache = [], [], [], []
        for num in range(self.num_layers):
            if self.use_dropout:
                if self.normalization == 'batchnorm':
                    if num != self.num_layers - 1:
                        out, cache = self.affine_batch_relu_forward(out, self.params['W' + str(num + 1)], self.params['b' + str(num + 1)], self.params['gamma' + str(num + 1)], self.params['beta' + str(num + 1)], self.bn_params[num])
                        fc_cache.append(cache[0])
                        relu_cache.append(cache[1])
                        bn_cache.append(cache[2])
                        out, cache = dropout_forward(out, self.dropout_param)
                        dropout_cache.append(cache)
                    else:
                        scores, fin_cache = affine_forward(out, self.params['W' + str(num + 1)] , self.params['b' + str(num + 1)])
                else:
                    if num != self.num_layers - 1:
                        out, cache = affine_relu_forward(out, self.params['W' + str(num + 1)], self.params['b' + str(num + 1)])
                        fc_cache.append(cache[0])
                        relu_cache.append(cache[1])
                        out, cache = dropout_forward(out, self.dropout_param)
                        dropout_cache.append(cache)
                    else:
                        scores, fin_cache = affine_forward(out, self.params['W' + str(num + 1)] , self.params['b' + str(num + 1)])
            else:
                if self.normalization == 'batchnorm':
                    if num != self.num_layers - 1:
                        out, cache = self.affine_batch_relu_forward(out, self.params['W' + str(num + 1)], self.params['b' + str(num + 1)], self.params['gamma' + str(num + 1)], self.params['beta' + str(num + 1)], self.bn_params[num])
                        fc_cache.append(cache[0])
                        relu_cache.append(cache[1])
                        bn_cache.append(cache[2])
                    else:
                        scores, fin_cache = affine_forward(out, self.params['W' + str(num + 1)] , self.params['b' + str(num + 1)])
                else:
                    if num != self.num_layers - 1:
                        out, cache = affine_relu_forward(out, self.params['W' + str(num + 1)], self.params['b' + str(num + 1)])
                        fc_cache.append(cache[0])
                        relu_cache.append(cache[1])
                    else:
                        scores, fin_cache = affine_forward(out, self.params['W' + str(num + 1)] , self.params['b' + str(num + 1)])
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        loss, softmax_grad = softmax_loss(scores, y)
        loss =  loss + self.reg * .5 * np.sum(fin_cache[1] * fin_cache[1])
        for cache in fc_cache:
             loss = loss + self.reg * .5 * np.sum(cache[1] * cache[1])
        dx, grads['W' + str(self.num_layers)], grads['b' + str(self.num_layers)] = affine_backward(softmax_grad, fin_cache)
        grads['W' + str(self.num_layers)] = grads['W' + str(self.num_layers)] + self.reg * 2 * .5 * fin_cache[1]
        for num in reversed(range(self.num_layers - 1)):
            if self.use_dropout:
                if self.normalization=='batchnorm':
                    dx = dropout_backward(dx, dropout_cache[num])
                    dx, grads['W'+ str(num + 1)], grads['b' + str(num + 1)], grads['gamma'+ str(num + 1)], grads['beta' + str(num + 1)] = self.affine_batch_relu_backward(dx, (fc_cache[num], relu_cache[num], bn_cache[num]))
                    grads['W'+ str(num + 1)] = grads['W'+ str(num + 1)] + self.reg * 2 * .5 * fc_cache[num][1]
                else:
                    dx = dropout_backward(dx, dropout_cache[num])
                    dx, grads['W'+ str(num + 1)], grads['b' + str(num + 1)] = affine_relu_backward(dx, (fc_cache[num], relu_cache[num]))
                    grads['W'+ str(num + 1)] = grads['W'+ str(num + 1)] + self.reg * 2 * .5 * fc_cache[num][1]
            else:
                if self.normalization=='batchnorm':
                    dx, grads['W'+ str(num + 1)], grads['b' + str(num + 1)], grads['gamma'+ str(num + 1)], grads['beta' + str(num + 1)] = self.affine_batch_relu_backward(dx, (fc_cache[num], relu_cache[num], bn_cache[num]))
                    grads['W'+ str(num + 1)] = grads['W'+ str(num + 1)] + self.reg * 2 * .5 * fc_cache[num][1]
                else:
                    dx, grads['W'+ str(num + 1)], grads['b' + str(num + 1)] = affine_relu_backward(dx, (fc_cache[num], relu_cache[num]))
                    grads['W'+ str(num + 1)] = grads['W'+ str(num + 1)] + self.reg * 2 * .5 * fc_cache[num][1]

        return loss, grads
    def affine_batch_relu_forward(self, x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that perorms an affine transform followed by a batch and a ReLU

        Inputs:
        - x: Input to the affine layer
        - w, b: Weights for the affine layer
        - gamma, beta: Weights for batchnorm
        -bn_param: the batchnorm params

        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = affine_forward(x, w, b)
        out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
        out, relu_cache = relu_forward(out)
        cache = (fc_cache, relu_cache, bn_cache)
        return out, cache
        
    def affine_batch_relu_backward(self, dout, cache):
        fc_cache, relu_cache, bn_cache = cache
        da = relu_backward(dout, relu_cache)
        d_bn, d_gamma, d_beta = batchnorm_backward(da, bn_cache)
        dx, dw, db = affine_backward(d_bn, fc_cache)
        return dx, dw, db, d_gamma, d_beta