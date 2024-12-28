import matplotlib.pyplot as plt    # To plot the data (visualization)
from Training_evaluation import evaluate

# To visualize model accuracy on validation data across training epochs
def plot_accuracies (history) :    
    # To extract the value of val_acc from within the history list
    accuracies = [x['val_acc' ] for x in history]
    
    # To draw a graph
    plt.plot (accuracies, '-x')
    
    # To make titles
    plt.xlabel ('epoch' )
    plt.ylabel ('accuracy')
    plt.title ('Accuracy vs. No. of epochs' )
    
    # Save plot by image
    plt.savefig('accuracy_plot.png')  
    plt.close()
    
# To visualize model loss on train data and validation data across training epochs
def plot_losses (history) :
    # To extract the value of train_loss from within the history list
    train_losses = [x.get ('train_loss' ) for x in history]
    # To extract the value of val_loss from within the history list
    val_losses = [x['val_loss' ] for x in history]
    
    # To draw a graph
    plt.plot (train_losses, '-bx' )  # blue
    plt.plot (val_losses, '-rx')     # red
    
    # To make titles
    plt.xlabel ('epoch')
    plt.ylabel ('loss')
    plt.legend(['Training', 'Validation'])  # Chart keys
    plt.title ('Loss vs. No. of epochs')
    
    # Save plot by image
    plt.savefig('loss_plot.png')
    plt.close()