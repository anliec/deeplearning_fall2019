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
    # print("z", np.isnan(z).any(), z.min(), z.max())
    z -= np.max(z, axis=1, keepdims=True)
    # print("z", np.isnan(z).any(), z.min(), z.max())
    e = np.exp(z)
    # print("e", np.isnan(e).any(), e.min(), e.max())
    s = np.sum(e, axis=1, keepdims=True)
    # print("s", np.isnan(s).any(), s.min(), s.max())
    p = e / s
    # print("p", np.isnan(p).any(), np.isnan(p).sum(), p.shape)
    t = p[range(len(y)), y]
    # print("t", np.isnan(t).any())
    w_norm = np.sum(W**2)
    # print("w_norm", np.isnan(w_norm).any())
    loss = - np.average(np.log(t)) + w_norm * reg
    assert not np.isnan(loss)
    # TODO: compute gradient decent
    dz = p
    dz[range(len(y)), y] -= 1
    dz /= len(y)
    dW = (X @ dz).T + W * (reg / w_norm)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
