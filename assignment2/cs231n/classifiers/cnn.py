from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        self.params['b1'] = np.zeros(num_filters)
        self.params['W1'] = weight_scale * np.random.randn(num_filters,input_dim[0],filter_size, filter_size)
        pad = (filter_size - 1) // 2
        out_conv_height = 1 + (input_dim[1] + 2 * pad - filter_size) 
        out_conv_width = 1 + (input_dim[2] + 2 * pad - filter_size)
        out_pool_height = int(1 + (out_conv_height - 2) / 2)
        out_pool_width = int(1 + (out_conv_width - 2) / 2)        
        self.params['W2'] = weight_scale * np.random.randn(num_filters * out_pool_height * out_pool_width, hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        grads = {}
        
        out, cache_crp = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, cache_rf = affine_relu_forward(out, W2, b2)
        scores, cache_f = affine_forward(out, W3, b3)
        if y is None:
            return scores
        loss, softmax_grad = softmax_loss(scores, y)
        loss = loss + self.reg * .5 * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))

        dx, grads['W3'], grads['b3'] = affine_backward(softmax_grad, cache_f)
        dx, grads['W2'], grads['b2'] = affine_relu_backward(dx, cache_rf)
        _, grads['W1'], grads['b1'] = conv_relu_pool_backward(dx, cache_crp)
        grads['W3'] = grads['W3'] + self.reg * .5 * 2 * W3
        grads['W2'] = grads['W2'] + self.reg * .5 * 2 * W2
        grads['W1'] = grads['W1'] + self.reg * .5 * 2 * W1

        return loss, grads
