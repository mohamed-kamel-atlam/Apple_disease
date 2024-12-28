import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def evaluate(model, val_dl):
    model.eval()  # Set model to evaluation mode
    outputs = []  # To store the results of validation steps

    with torch.no_grad():
        for batch in val_dl:
            result = model.validation_step(batch)  # Use validation_step
            outputs.append(result)  # Collect batch results
    
    # Aggregate the results using validation_epoch_end
    return model.validation_epoch_end(outputs)


# A function to train the model over several epochs
def fit(epochs, lr, model, train_dl, val_dl, opt_func):
    optimizer = opt_func(model.parameters(), lr)  # Optimizer
    history = []  # To store results for each epoch

    for epoch in tqdm(range(epochs)):
        # Model training
        model.train()  # Set model to training mode
        train_losses = []
        
        for batch in train_dl:
            loss = model.training_step(batch)  # Use training_step
            train_losses.append(loss.item())  # Collect loss
            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters
            optimizer.zero_grad()  # Reset gradients

        # Validation
        result = evaluate(model, val_dl)  # Evaluate on validation set
        result['train_loss'] = torch.tensor(train_losses).mean().item()  # Average training loss
        history.append(result)  # Save epoch results

        # Print metrics for the epoch
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {result['train_loss']:.4f}, "
              f"Val Loss: {result['val_loss']:.4f}, Val Accuracy: {result['val_acc']:.4f}")
        print(f"Precision: {result['precision']:.4f}, Recall: {result['recall']:.4f}, F1-Score: {result['f1']:.4f}")

    return history  # Return training history


# Function to plot metrics (precision, recall, F1-score) over epochs
def plot_metrics(history):
    # Extract metrics for all epochs
    precision = [epoch_data['precision'] for epoch_data in history]
    recall = [epoch_data['recall'] for epoch_data in history]
    f1 = [epoch_data['f1'] for epoch_data in history]

    # Define epochs
    epochs = range(1, len(history) + 1)

    plt.figure(figsize=(12, 6))
    
    # Plot metrics
    plt.plot(epochs, precision, label='Precision', marker='o')
    plt.plot(epochs, recall, label='Recall', marker='o')
    plt.plot(epochs, f1, label='F1-Score', marker='o')
    
    # Add titles, labels, and legend
    plt.title('Precision, Recall, and F1-Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()
    plt.grid(True)
    
    # Save and close the plot
    plt.savefig('prc_rec_f1_epoch.png')
    plt.close()

def plot_confusion_matrix(model, val_dl, classes):
    model.eval()  # Set model to evaluation mode
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in val_dl:
            inputs, targets = batch
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            y_true.extend(targets.cpu().numpy())  # True labels
            y_pred.extend(predicted.cpu().numpy())  # Predicted labels

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')  # Save the plot as a PNG file
    plt.close()

