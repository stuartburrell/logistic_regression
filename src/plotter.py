import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

def plot_losses(name, train_losses, val_losses):
 
    '''
    Training loop implementing L2-regularized SGD

    Args:
        name : filename for saved png file
        train_losses : list of training losses
        val_losses : list of validation losses

    Returns: 
        None
    '''
  
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_title('Learning curves', fontsize=18, pad=15, fontweight='bold')
    ax.plot(range(1, len(train_losses) + 1), train_losses, label='Training set')
    ax.plot(range(1, len(val_losses) + 1), val_losses, label='Validation set')
    ax.set_ylabel('Loss', labelpad=15, fontsize=18)
    ax.set_xlabel('Epoch', labelpad=15, fontsize=18)
    ax.legend(fontsize=18)
    ax.tick_params(labelsize=15)
    fig.savefig('plots/' + name + '.png')