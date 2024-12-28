import os    # To handle files                     
import torchvision.transforms as T   # To provide data transformation operations
from torchvision.datasets import ImageFolder   # To load a data set stored in folders
from torch.utils.data.dataloader import DataLoader   # To load data efficiently

def pre_processing():
    
    # Define a datapath variable
    data_dir = 'New Plant Diseases Dataset(Augmented)'
    print(os.listdir(data_dir))
    
    # Extract category names 
    classes = os.listdir (data_dir + "/train") 
    
    # General settings
    image_size = 32     # Images reshaping (32 * 32)
    batch_size = 128    # The size of each batch of images that is passed to the model during training
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))   # Normalization => (Mean(RGB) , Std(RGB))
    
    # Augmentation
    # To apply a series of transformations to images (train and valid)
    train_transforms = T.Compose([
        T.Resize(image_size),          # Usually resize images to (32 * 32)
        T.CenterCrop(image_size),      # Cut the image from the center to fit the desired dimensions
        T.RandomHorizontalFlip(),      # Make a random horizontal flip of the image
        T.RandomRotation(30),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(*stats)            # Normalize data using statistical values
    ])

    val_transforms = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats)
    ])
    
    # Definition of data sets
    # Load datasets
    train_ds = ImageFolder(data_dir + "/train", transform=train_transforms)
    val_ds = ImageFolder(data_dir + "/valid", transform=val_transforms)
    print(f"Classes: {train_ds.classes}")
    
    # Get the total number of data samples in the training and validation datasets
    total_train_samples = len(train_ds)
    total_val_samples = len(val_ds)

    # Print the total counts
    print(f"Total training samples: {total_train_samples}")
    print(f"Total validation samples: {total_val_samples}")
    
    # Definition of download tools
    train_dl = DataLoader (train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    val_dl = DataLoader (val_ds, batch_size*2, num_workers=4, pin_memory=True) 
    
    return data_dir , train_dl , val_dl , classes , val_transforms