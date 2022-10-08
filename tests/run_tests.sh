#!/bin/bash

NUM_EPOCHS=100
LR=0.0001
GAMMA=0.0001
NORMALIZE=True

echo ---------------------------;
echo -- Testing configuration --;
echo Number of epochs: $NUM_EPOCHS;
echo Learning rate: $LR;
echo Gamma: $GAMMA;
echo Normalize: $NORMALIZE;
echo ---------------------------;

echo ------------------------------------------;
echo Testing on fraud dataset ...
echo ------------------------------------------;

python tests/test.py data/fraud.csv --num_epochs $NUM_EPOCHS --lr $LR \
                                     --gamma $GAMMA --normalize $NORMALIZE

echo ------------------------------------------;
echo Testing on UCI Income ...
echo ------------------------------------------;

python tests/test.py data/income.csv --num_epochs $NUM_EPOCHS --lr $LR \
                                     --gamma $GAMMA --normalize $NORMALIZE

echo ------------------------------------------;
echo Testing main.py script ...
echo ------------------------------------------;

python main.py data/fraud.csv --num_epochs $NUM_EPOCHS --lr $LR \
                             --gamma $GAMMA --normalize $NORMALIZE
