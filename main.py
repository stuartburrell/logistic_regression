'''
A script to train and evaluate logistic regression on a tabular dataset
with continuous features.

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
'''

from src.config import load_config
from src.datasets import CSVDataset
from src.logistic_regression import LogisticRegression
from src.plotter import plot_losses

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
                                            save_path=config.save_path,
                                            learning_rate=config.lr, 
                                            gamma=config.gamma,
                                            num_epochs=config.num_epochs)
    

    # plot learning curves to check for convergence and/or overfitting
    plot_losses('learning_curves_{}'.format(config.data.split('/')[1].split('.')[0]),
                training_losses, val_losses)

    # report model performance on test data
    print('\n---Classification Report---\n\n', model.evaluate(x_test, y_test))
