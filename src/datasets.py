import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

class CSVDataset():
    
    '''
    Generic CSV dataset class for loading and preprocessing
    '''
    
    def __init__(self, filepath, normalize):
        
        # load data
        data = pd.read_csv(filepath, dtype=float)
        data.columns = list(range(len(data.columns)))
        
        # remove rows with missing values
        data.dropna(axis=0)
        
        # store dataset metadata
        self.n_rows = data.shape[0]
        self.n_features = data.shape[1] - 1
        
        # extract features and targets
        self.targets = data[self.n_features].astype(float).to_numpy()
        self.features = data.drop(columns=self.n_features).astype(float).to_numpy()
        
        # normalize features if desired
        if normalize: self.features = (self.features - np.mean(self.features)) / np.std(self.features)

    def get_splits(self, seed=42):
     
        '''
        Helper function to generate train/val/test split
        
        Args:
            seed : random seed for reproducibility
        
        Returns:
            x_train : training feature matrix
            y_train : training binary target vector
            x_val : validation feature matrix
            y_val : validation binary target vector
            x_test : testing feature matrix
            y_test : testing binary target vector
        '''
      
        x_train, x_rest, y_train, y_rest = train_test_split(self.features, 
                                                            self.targets, 
                                                            test_size=0.2, 
                                                            random_state=seed)

        x_val, x_test, y_val, y_test = train_test_split(x_rest, 
                                                        y_rest, 
                                                        test_size=0.5, 
                                                        random_state=seed)

        return x_train, y_train, x_val, y_val, x_test, y_test
       
       
       
                           
    
