import argparse

def load_config():

    '''
    Helper function to parse and load command line arguments
    
    Returns:
        args : namespace of argument values
    '''
  
    parser = argparse.ArgumentParser()
    parser.add_argument("data",
                        help="filepath to data CSV file")
    parser.add_argument("--save_path", metavar='\b',
                        help="path to store model weights, default=None", 
                        default=None, type=str)
    parser.add_argument("--num_epochs", metavar='\b',
                        help="number of training epochs, default=300", 
                        default=300, type=int)
    parser.add_argument("--lr", metavar='\b',
                        help="learning rate for SGD, default=0.0001", 
                        default=0.0001, type=float)
    parser.add_argument("--gamma", metavar='\b',
                        help="regularisation penalty coefficient, default=0.0001", 
                        default=0.0001, type=float)
    parser.add_argument("--normalize", metavar='\b',
                        help="option to normalize input features, default=True", 
                        default=True, type=bool)
    args = parser.parse_args()
    
    return args