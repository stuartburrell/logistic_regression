# Logistic regression from scratch

This repository provides a simple implementation of logistic regression. 
Training is performed using SGD with L2-regularisation.

## Contents

- main.py : example training and evaluation script
- requirements.txt : required python packages
- source/
  * config.py : function too parse command line arguments
  * datasets.py : class to load and preprocess tabular data
  * logistic_regression.py : model class for logistic regression
  * plotter.py : function to plot learning curves
- plots/ : directory to save figures
- data/ : directory to store data
- weights/ : directory to store trained weights
- tests/
  * run_tests.sh : bash script for running a few simple tests on two datasets
  * test.py : script comparing our implementation against the sklearn equivalent

## Getting started

To install the required packages, navigate to this directory and run
```
pip install -r ./requirements.txt
```

To train and evaluate on a fraud dataset, run

```
python main.py data/fraud.csv --save_path './weights/fraud_demo.pkl' --num_epochs 300 \
                             --lr 0.0001 --gamma 0.0001 --normalize True
```

To train and evalute on an further example dataset (UCI Income), run

```
python main.py data/income.csv --save_path './weights/income_demo.pkl' --num_epochs 300 \
                               --lr 0.0001 --gamma 0.0001 --normalize False
```

To obtain help, use

```
python main.py --help
```
```
usage: main.py [-h] [--save_path] [--num_epochs] [--lr] [--gamma] [--normalize] data

positional arguments:
  data            filepath to data CSV file

optional arguments:
  -h, --help      show this help message and exit
  --save_path   path to store model weights, default=None
  --num_epochs  number of training epochs, default=300
  --lr          learning rate for SGD, default=0.0001
  --gamma       regularisation penalty coefficient, default=0.0001
  --normalize   option to normalize input features, default=True
```

## Tests

To verify the implementation against the sklearn version, run

```
chmod +x ./tests/run_tests.sh
./tests/run_tests.sh
```

from the main directory.

## Extensions

This is a simple example of logistic regression. Extensions could include exploratory data analysis to engineer alternative
features, the use of dimensionality reduction or feature selection (e.g. PCA), or fine-tuning hyperparameters such as the 
regularisation penalty using cross-validation. Deeper networks could also likely improve performance.
