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
    z = W @ X
    z -= np.max(z)
    e = np.exp(z)
    s = np.sum(e, axis=0)
    p = np.divide(e, s)
    t = p[y, np.arange(len(y))]
    w_norm = np.linalg.norm(W, 2)
    loss = - np.average(np.log(t)) + w_norm * reg
    # TODO: compute gradient decent
    y_categorical = np.zeros_like(p)
    y_categorical[y, np.arange(len(y))] = 1
    dW = np.matmul((p - y_categorical), X.T) + reg / w_norm * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
