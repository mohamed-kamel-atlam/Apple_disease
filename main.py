"""
This script performs the following tasks:

1. Pre-processes the dataset for training and validation.
2. Defines and initializes a Convolutional Neural Network (CNN) model.
3. Manages device allocation (CPU/GPU) for data and model.
4. Trains the model, evaluates it, and saves the trained parameters.
5. Visualizes performance metrics (accuracy and loss) during training.
6. Tests the model on unseen test data and evaluates its performance.
7. Saves plots of results and model outputs.
"""

if __name__ == "__main__":
    import torch
    from Pre_processing import pre_processing
    from CNN_Model import CnnModel
    from device_manager import get_default_device
    from device_manager import DeviceDataLoader
    from device_manager import to_device
    from Training_evaluation import fit
    from Training_evaluation import evaluate
    import time
    from plot_acc_loss import plot_accuracies
    from plot_acc_loss import plot_losses
    from Training_evaluation import plot_metrics
    from torchvision.datasets import ImageFolder             
    from Test_Data import denorm
    from Test_Data import predict_image
    from Test_Data import evaluate_test
    import matplotlib.pyplot as plt    
    from torch.utils.data.dataloader import DataLoader 
    from Training_evaluation import plot_confusion_matrix    
    
    
    #____________________ Pre pocessing ________________________
    
    data_dir , train_dl, val_dl, classes , val_transforms = pre_processing()

    #____________________ Create the model _____________________
    
    model = CnnModel()
    print(model)
    
    #____________________ device manager _______________________
    
    # Find out which device is being used (cuda or cpu)
    device = get_default_device ()
    device
    
    # Data batches are automatically transferred to the selected device when duplicated
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)

    # The model is set up to execute on the same machine to which the training or validation data will be transferred
    to_device (model, device)
    
    # The model is prepared to work on the selected device
    model = to_device(CnnModel() , device)

    #____________________ Training Data ___________________
    # Training Settings
    num_epochs = 8
    opt_func = torch.optim.AdamW
    lr = 0.001

    # Start training and measure time
    start_time = time.time()
    history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    end_time = time.time()

    # Time taken
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Evaluate model performance on validation data
    evaluate(model , val_dl)
    
    # Save prameters after end model
    torch.save(model.state_dict(), 'modelAppleFile1.pth')

    #____________________ plot acc&loss __________________
    
    # plot accuracy (Validation)
    plot_accuracies(history)
    # plot losses (Train , Validation)
    plot_losses(history)
    # plot recoll , perction , F1
    plot_metrics(history)
    # plot confusion matrix
    plot_confusion_matrix(model, val_dl, classes)

    #______________________ Test Data______________________
    
    test_ds = ImageFolder(data_dir + "/test", transform=val_transforms)
    img, label = test_ds[42]
    plt.imshow(denorm(img).permute(1, 2, 0))
    print("Label:", test_ds.classes[label], ", Predicted:", predict_image(img, model))
    
    plt.savefig('test.png') 
    plt.close()
    
    # Convert test data to DataLoader
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    # Evaluate the model on test data
    test_loss, test_accuracy = evaluate_test(model, test_loader, test_ds.classes)

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4%}")

    # Collect test losses and accuracies for plotting 
    test_losses = [test_loss] 
    test_accuracies = [test_accuracy] 
    
    # Plotting test losses and accuracies 
    plt.plot(test_losses, '-bx') # Blue for losses 
    plt.plot(test_accuracies, '-rx') # Red for accuracies 
    
    # Add titles and labels 
    plt.xlabel('epoch') 
    plt.ylabel('loss/accuracy') 
    plt.legend(['Test Losses', 'Test Accuracy']) 
    
    # Chart keys 
    plt.title('Loss vs. No. of epochs') 
    
    # Save plot as images 
    plt.savefig('loss_acc_test.png') 