#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python3 -u train.py \
    --model convnet \
    --kernel-size 5 \
    --hidden-dim 16 \
    --epochs 10 \
    --weight-decay 0.0001 \
    --momentum 0.6 \
    --batch-size 1024 \
    --lr 0.00001 | tee convnet.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
