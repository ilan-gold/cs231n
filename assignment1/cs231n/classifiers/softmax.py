import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  # Go over all examples
  for ex in range(num_train):
      scores_denom = 0
      scores = []
      # Get raw scores and normalize
      for cls in range(num_classes):
        scores += [np.dot(W[:,cls], X[ex,:])]
      scores -= np.max(scores)
      # Create denominator for softmax
      for score in scores:
          scores_denom += np.exp(score)
      # Create vector of all softmax for all classes
      softmax_vec = np.exp(scores) / scores_denom
      softmax = softmax_vec[y[ex]]
      # Compute loss and gradient
      loss += -1 * np.log(softmax)
      dW += ((softmax_vec - np.array(list(range(num_classes)) == y[ex]).astype(int))[:,np.newaxis] * X[ex, :]).transpose()
  # Cleanup with regularitzaiton/averaging
  loss = (1 / num_train)  * loss + reg * np.sum(np.dot(W, W.transpose()))
  dW = (1 / num_train) * dW
  dW += 2 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = np.dot(X, W)
  scores = (scores.transpose() - np.max(scores, axis = 1)).transpose()
  # Exponentiated scores of shape N x C
  scores = np.exp(scores)
  # Create vector of all softmax for all classes of shape also N x C
  softmax_vec = (scores.transpose() / np.sum(scores, axis = 1)).transpose()
  # Create correct class softmax of shape N x 1
  softmax = softmax_vec[np.arange(len(softmax_vec)), y]
  # Compute loss and gradient
  loss += np.sum(-1 * np.log(softmax))
  # Create one-hot-vec encoding
  nb_classes = W.shape[1]
  targets = np.array([y]).reshape(-1)
  one_hot_targets = np.eye(nb_classes)[targets]
  # Use N x C one hot vec and softmax to multiply by N x D example matrix to get D x C dW
  dW += (np.dot((softmax_vec - one_hot_targets).transpose(),X)).transpose()
  # Cleanup with regularitzaiton/averaging
  loss = (1 / num_train)  * loss + reg * np.sum(np.dot(W, W.transpose()))
  dW = (1 / num_train) * dW
  dW += 2 * reg * W

  return loss, dW

