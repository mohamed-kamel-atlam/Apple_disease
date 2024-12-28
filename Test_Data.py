import torch
from Pre_processing import pre_processing
import torch.nn as nn            


# Denormalization function
def denorm(img_tensors):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)   
    
    return img_tensors * std + mean  
# It returns normalized images to their original form using mean and std

_ , _ , _ , classes , val_transforms = pre_processing()
def predict_image(img, model):
    # Transfer the image to the device used (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert the image to Batch and send it to the form
    img = img.unsqueeze(0).to(device) # Add after Batch
    model.eval() 
    with torch.no_grad():  # Disable calculation of gradients
        outputs = model(img)
        _, predicted = torch.max(outputs, dim=1) # Choose the classification with the highest probability
    # Returns the expected taxonomy name based on the list
    return classes[predicted.item()]

def evaluate_test(model, test_loader, classes):
    model.eval() # Put the form into evaluation mode
    total_correct = 0
    total_samples = 0
    total_loss = 0
    
    # Choose a loss function
    loss_fn = nn.CrossEntropyLoss()
    
    with torch.no_grad():  # Disable gradients to improve performance
        for images, labels in test_loader:
            
            # Pass data through the form
            outputs = model(images)
            
            # Loss calculation
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * images.size(0)  # Collect losses for each image
            
            # Calculate correct predictions
            _, preds = torch.max(outputs, dim=1)  # Select the highest probability classifications
            total_correct += torch.sum(preds == labels).item()  # Number of correct classifications
            total_samples += labels.size(0)  # Update the total number of the sample
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return avg_loss, accuracy

