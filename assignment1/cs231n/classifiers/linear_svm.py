import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  first = True
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in range(num_classes):
      if first:
          dW[:,j] += reg * 2 * W[:,j]
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] +=  (1 / 500) * X[i]
        count += 1
    dW[:,y[i]] += (1 / 500) * -1 * count *  X[i]
    first = False

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = X.dot(W)
  dW += reg * 2 * W
  #Create margin
  margin = (scores.transpose() - scores[np.arange(len(scores)), y]).transpose() + 1
  #Make correct class margin to be 0
  margin[np.arange(len(scores)), y] = 0
  #Update loss and gradient
  loss += np.sum(margin[(margin > 0)])
  dW += np.dot((margin > 0).transpose(), X).transpose()
  #Create number of times margin is greater than 0
  counts_vec = np.sum((margin > 0), axis = 1)
  #Multiply this by X row-by-row-by-count to get weighted train matrix
  counts_by_train_mat = counts_vec[:, np.newaxis] * X
  #Create a one-hot-vec encoding of the correct classes to get only the necessary updates
  nb_classes = 10
  targets = np.array([y]).reshape(-1)
  one_hot_targets = np.eye(nb_classes)[targets]
  #Get the necessary updates
  update_mat = np.dot(one_hot_targets.transpose(), counts_by_train_mat).transpose()
  #Update
  dW += -1 * update_mat
  dW /= num_train
  loss /= num_train
  loss += reg * np.sum(W * W)

  return loss, dW
