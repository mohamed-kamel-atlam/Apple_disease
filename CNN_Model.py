import torch                       
import torch.nn as nn              # for designing neural networks
# Compute precision, recall, and F1 score
from sklearn.metrics import precision_score, recall_score, f1_score

# Number of Convolutional Layers ==> 6
# After 2 convolution layer, pooling works to reduce the size of the images
# Then, Flatten the data to a single column (Vector)
# In the end, the output is 4 classes (types of apple diseases)

class CnnModel(nn.Module):
    # Constructor
    def __init__(self):
        super(CnnModel, self).__init__()
        # Define the layers using nn.Sequential
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input: 3 x 32 x 32 -> Output: 32 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output: 64 x 32 x 32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: 128 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # Output: 128 x 16 x 16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Output: 256 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Output: 256 x 8 x 8
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x 4 x 4
        )

        # Dropout layer to avoid overfitting
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),  # Input: 256*4*4 -> Output: 512
            nn.ReLU(),
            nn.Linear(512, 4)  # Output: 4 classes
        )

    # Define forward function
    def forward(self, x):
        x = self.features(x)  # Pass input through feature extractor
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(x)  # Apply dropout (avoid overfitting)
        x = self.classifier(x)  # Pass through classifier
        return x

    # Training function for each batch
    def training_step(self, batch):
        images, labels = batch
        outputs = self(images)  # Model predictions
        loss = nn.CrossEntropyLoss()(outputs, labels)  # Loss calculation
        return loss

    # Validation step to compute loss and metrics for a batch
    def validation_step(self, batch):
        images, labels = batch
        outputs = self(images)  # Generate predictions
        loss = nn.CrossEntropyLoss()(outputs, labels)  # Calculate loss
        _, preds = torch.max(outputs, dim=1)  # Get class predictions
        correct = (preds == labels).sum().item()
        return {
            'val_loss': loss.item(),
            'val_correct': correct,
            'val_total': len(labels),
            'y_true': labels.cpu(),
            'y_pred': preds.cpu(),
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        total_correct = sum(x['val_correct'] for x in outputs)
        total_samples = sum(x['val_total'] for x in outputs)
        val_acc = total_correct / total_samples

        # Flatten predictions and true labels
        y_true = torch.cat([x['y_true'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])

        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        return {
            'val_loss': avg_loss.item(),
            'val_acc': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
