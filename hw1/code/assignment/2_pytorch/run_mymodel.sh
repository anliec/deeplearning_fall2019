#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python3 -u train.py \
    --model mymodel \
    --kernel-size 1 \
    --hidden-dim 20 \
    --epochs 1 \
    --weight-decay 0.0003 \
    --momentum 0.9 \
    --batch-size 96 \
    --lr 0.025 \
    --model_state_dict hw_cifar10_state.pt | tee mymodel.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
