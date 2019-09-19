#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python3 -u train.py \
    --model softmax \
    --epochs 5 \
    --weight-decay 0.0001 \
    --momentum 0.9 \
    --batch-size 2048 \
    --lr 0.001 | tee softmax.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
