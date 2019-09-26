import numpy as np
from random import shuffle


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs:
    - W: C x D array of weights
    - X: D x N array of data. Data are D-dimensional columns
    - y: 1-dimensional array of length N with labels 0...K-1, for K classes
    - reg: (float) regularization strength
    Returns:
    a tuple of:
    - loss as single float
    - gradient with respect to weights W, an array of same size as W
    """
    # Initialize the loss and gradient to zero.
    # loss = 0.0
    # dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    z = X.T @ W.T
    z -= np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    s = np.sum(e, axis=1, keepdims=True)
    p = e / s
    t = p[range(len(y)), y]
    w_norm = np.sum(W**2)
    loss = - np.average(np.log(t)) + w_norm * reg
    assert not np.isnan(loss)
    dz = p
    dz[range(len(y)), y] -= 1.0
    # dz /= len(y)
    # dW = (X @ dz).T + W * (reg / w_norm)
    dW = (X @ dz).T / len(y) + W * reg
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
