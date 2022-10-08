'''
A script to test our implementation of logistic regression against the standard sklearn version.

usage: test.py [-h] [--num_epochs] [--lr] [--gamma] [--normalize] data

positional arguments:
  data            filepath to data CSV file

optional arguments:
  -h, --help      show this help message and exit
  --num_epochs    number of training epochs, default=100
  --lr            learning rate for SGD, default=0.0001
  --gamma         regularisation penalty coefficient, default=0.0001
  --normalize     option to normalize input features, default=True
'''

import sys
import os
 
current = os.path.dirname(os.path.realpath(__file__))
parent= os.path.dirname(current)
sys.path.append(parent)

from src.config import load_config
from src.datasets import CSVDataset
from src.logistic_regression import LogisticRegression
from src.plotter import plot_losses
from sklearn.metrics import classification_report

if __name__ == "__main__":
 
    # load experiment configuration
    config = load_config()

     # load dataset
    dataset = CSVDataset(filepath=config.data, normalize=config.normalize)

    # initialise model
    model = LogisticRegression()

    # obtain train/val/test split 
    x_train, y_train, x_val, y_val, x_test, y_test = dataset.get_splits()
    

    # fit model with training data
    print('\nTraining logistic regression model...\n')
    training_losses, val_losses = model.fit(x_train,
                                            y_train, 
                                            x_val,
                                            y_val,
                                            save_path=None,
                                            learning_rate=config.lr, 
                                            gamma=config.gamma,
                                            num_epochs=config.num_epochs)

    # report model performance on test data
    print('\n---Classification Report (our model)---\n\n', model.evaluate(x_test, y_test))
    
    # overwrite model class with sklearn implementation
    from sklearn.linear_model import LogisticRegression
    
    # train sklearn implementation
    model = LogisticRegression(random_state=42, max_iter=10000).fit(x_train, y_train)
                        
    # report sklearn model performance on test data
    print('\n---Classification Report (sklearn model)---\n\n', classification_report(y_test, model.predict(x_test)))  
