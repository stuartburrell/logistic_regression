import numpy as np
import pickle

from tqdm import tqdm
from sklearn.metrics import classification_report

class LogisticRegression():
 
    '''
    A model class for logistic regression with L2-regularized SGD training
    '''
  
    def __init__(self, weights_path=None):
    
        '''
        Initialise  model class
        
        Args:
            weights_path : filepath to restore pre-trained weights (and bias)
            
        Returns: None
        '''
     
        if weights_path:
            self.weights, self.bias = pickle.load(open(weights_path, 'rb'))
        else:
            self.weights = None
            self.bias = 0

    def process_epoch(self, features, targets, gamma, learning_rate, val=False):
     
        '''
        Helper function to process data for one epoch and return loss - 
        if in validation mode parameters are frozen
        
        Args:
            features : matrix of float training features
            targets : binary vector of training targets
            learning_rate : step-size for SGD updates
            gamma : L2 regularisation coefficient
            
        Returns: 
            losses : mean loss over epoch
        '''
      
        losses = []
        for sample, target in zip(features, targets):  
            # compute prediction with current parameters
            pred = 1 / (1 + np.exp(-(np.dot(self.weights, sample) + self.bias)))
            losses.append(-pred * np.log(target + 10**-8) - (1 - pred) * np.log(1 - target + 10**-8))
            
            # if not in validation mode update weights
            if not val:
                # compute binary cross-entropy loss derivatives w.r.t. weights and bias
                l2_reg_penalty = self.weights * gamma / features.shape[0]
                d_weights = -sample * (target - pred - l2_reg_penalty)
                d_bias = -target + pred

                # update weights with SGD step
                self.weights = self.weights - learning_rate * d_weights
                self.bias = self.bias - learning_rate * d_bias

        return np.mean(losses)

    def fit(self, 
            train_features, 
            train_targets, 
            val_features,
            val_targets,
            save_path='./weights/demo.pkl',
            learning_rate=0.001,
            gamma=0.0001,
            num_epochs=100,
            seed=42):
     
        '''
        Training loop implementing L2-regularized SGD
        
        Args:
            train_features : matrix of float training features
            train_targets : binary vector of training targets
            val_features : matrix of float validation features
            val_targets : binary vector of validation targets
            save_dir : directory to save trained model weights
            learning_rate : step-size for SGD updates
            gamma : L2 regularisation coefficient
            num_epochs : number of training epochs
            seed : random seed for reproducibility
            
        Returns: 
            training_losses : mean loss over each training set each epoch
            val_losses : mean loss over each validation set each epoch
        '''
        
        # initialise appropiate number of weights
        self.weights = np.zeros(shape=train_features.shape[1])
       
        # train with SGD and compute losses
        training_losses, val_losses = [], []
        with tqdm(range(num_epochs)) as epochs:
            for epoch in epochs:
                 training_losses.append(self.process_epoch(train_features, 
                                                           train_targets, 
                                                           gamma, 
                                                           learning_rate, 
                                                           val=False))
                 val_losses.append(self.process_epoch(val_features, 
                                                      val_targets, 
                                                      gamma, 
                                                      learning_rate, 
                                                      val=True))
                 if epoch % 5 == 0: 
                     epochs.set_postfix(train_loss=training_losses[-1], val_loss=val_losses[-1])
                                                       
        # save trained model weights    
        if save_path:
            pickle.dump((self.weights, self.bias), open(save_path, 'wb'))

        return training_losses, val_losses
        
    def predict(self, features):
     
        '''
        Generate predictions
        
        Args:
            features : matrix of float training features
            
        Returns: 
            preds : binary vector of class predictions
        '''
      
        preds = (1 / (1 + np.exp(-(np.sum(self.weights * features, axis=1) + self.bias)))) > 0.5
       
        return preds
    
    def evaluate(self, features, targets):
     
        '''
        Evaluate performance of the model
        
        Args:
            features : matrix of float training features
            targets : binary vector of class targets 
            
        Returns: 
            report : sklearn classification report (recall, precision, f1, accuracy, ...)
        '''

        report = classification_report(targets, self.predict(features))  
       
        return report