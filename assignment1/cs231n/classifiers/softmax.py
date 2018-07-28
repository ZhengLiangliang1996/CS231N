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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  for i in range(num_train): # in each loop of train,get the loss of every loop
      score = X[i].dot(W)
      f = score - max(score)#according to the cs231n linear_classify notes
      loss_i = - f[y[i]] + np.log(sum(np.exp(f)))
      loss = loss_i + loss 
      for j in range(num_classes): # in each loop of classes, get the dW
        softmax = np.exp(f[j]) / sum(np.exp(f))
        if j == y[i]:
            dW[:,j] += (-1 + softmax) * X[i]
        else:
            dW[:,j] += softmax * X[i]
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W) # add the regularization term
  dW = dW/num_train + reg * W
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  score = X.dot(W)
  f = score - np.max(score, axis = 1).reshape(-1,1) # get max in every row and stretch
  softmax = np.exp(f) / np.sum(np.exp(f),axis = 1).reshape(-1,1)
  loss = -np.sum(np.log(softmax[range(num_train), list(y)]))# question!!!!
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W) # add the regularization term
    
  dS = softmax.copy()
  dS[range(num_train),list(y)] += -1
  dW = (X.T).dot(dS)
  dW = dW/num_train+ reg * W
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

