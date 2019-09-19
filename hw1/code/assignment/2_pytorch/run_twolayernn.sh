#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python3 -u train.py \
    --model twolayernn \
    --hidden-dim 256 \
    --epochs 30 \
    --weight-decay 0.001 \
    --momentum 0.6 \
    --batch-size 2048 \
    --lr 0.001 | tee twolayernn.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
