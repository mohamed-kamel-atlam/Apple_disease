import streamlit as st 
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import torch
import torchvision.transforms as T   # To provide data transformation operations
from CNN_Model import CnnModel
from Pre_processing import pre_processing
from torchvision.datasets import ImageFolder        
from torch.utils.data.dataloader import DataLoader 
from Test_Data import evaluate_test     


# Load your trained model
@st.cache_resource
def load_image(image_file):
    img = Image.open(image_file)
    return img

st.set_page_config(
    page_title="Apple Leaf Disease Detection",
    page_icon = ":apple:",
    initial_sidebar_state='auto'
)

st.title("Apple Disease Prediction")
st.sidebar.image("AppleDisease.jpeg")
st.sidebar.info("Make a deep learning model that can detect plant diseases (especially apple diseases) from images.")
    

image_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

# Initialize and load the model
model = CnnModel()
model.load_state_dict(torch.load('modelAppleFile1.pth')) 
model.eval()


# Ensure classes are loaded correctly from pre_processing
data_dir, _, _, classes, val_transforms = pre_processing()  
# st.write(f"Loaded classes: {classes}")  # Display loaded classes for debugging

# Determine the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)  # Move the model to the appropriate device

# Prediction function
def predict_image(img, model, device):
    model.eval()  # Ensure the model is in evaluation mode
    img = img.to(device)  # Move image tensor to the correct device
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(img)
        _, predicted = torch.max(outputs, dim=1)  # Choose the class with the highest probability
    return predicted.item()

# Disease details function
def get_disease_details(class_index):
    details = {
        0: {
            "name": "Apple Scab",
            "description": "A fungal disease causing dark, scabby spots on leaves and fruit.",
            "treatment": "Use fungicides such as Captan or Mancozeb. Remove infected leaves and fruit from the orchard.",
            "precautions": "Plant resistant apple varieties and ensure proper airflow between trees."
        },
        1: {
            "name": "Black Rot",
            "description": "A fungal disease leading to dark, sunken spots on fruit and leaf edges.",
            "treatment": "Prune infected branches and apply fungicides like Thiophanate-methyl.",
            "precautions": "Avoid wounding trees and remove fallen leaves and fruit promptly."
        },
        2: {
            "name": "Cedar Apple Rust",
            "description": "A fungal disease causing yellow-orange spots on leaves and fruit.",
            "treatment": "Apply fungicides like Myclobutanil. Remove nearby juniper trees, which are alternate hosts.",
            "precautions": "Monitor weather conditions and spray fungicides preventively."
        },
        3: {
            "name": "Healthy",
            "description": "The apple is healthy with no signs of disease.",
            "treatment": "No treatment necessary.",
            "precautions": "Continue maintaining good orchard management practices."
        }
    }
    return details.get(class_index, {"name": "Unknown", "description": "Unknown disease class."})

if image_file is not None:
    # Display the uploaded image
    st.image(load_image(image_file), width=250)
    
    # Load the image and resize it to the expected size (32x32 in this case)
    image = Image.open(image_file)
    image = image.resize((32, 32))
    
    # Normalization => (Mean(RGB) , Std(RGB))
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(*stats)
    ])
    
    # Apply the transformatio
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Print the shape of the tensor for debugging
    print("Shape of input tensor:", image_tensor.shape)  # Expected shape: [1, 3, 32, 32]

    # Get the predicted class index from the model
    predicted_class_index = predict_image(image_tensor, model, device)
    
    # Get disease details
    details = get_disease_details(predicted_class_index)
    
    # -----------------Get acc & loss for test ------------------------------
    test_ds = ImageFolder(data_dir + "/test", transform=val_transforms)
    
    # Convert test data to DataLoader
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    # Evaluate the model on test data
    test_loss, test_accuracy = evaluate_test(model, test_loader, test_ds.classes)

    st.sidebar.write("")
    st.sidebar.success(f"Test Loss: {test_loss:.4f}")
    st.sidebar.success(f"Test Accuracy: {test_accuracy:.4%}")
    st.sidebar.write("")
    
    # Display the predicted class, description, and drag details
    if details['name']== "Healthy":
      st.success(f"Predicted Disease: {details['name']}")
      st.balloons()
    else :
       st.warning(f"Predicted Disease: {details['name']}")

    st.info(f"Disease Description: {details['description']}")
 
    st.info(f"treatment: {details['treatment']}")    # Treatment
   
    st.info(f"precautions: {details['precautions']}")  # Precautions
    
    # Visualization
    visualization_choice = st.sidebar.selectbox(
        "Choose a Visualization:",
        [
            "None",
            "Loss Over Epochs",
            "Accuracy Over Epochs",
            "Precision, Recall, F1-Score Over Epochs",
            "confusion_matrix"
        ],
        key="visualization_1"
    )

    # Display the selected visualization
    st.subheader("Training and Validation Visualizations")
    st.write("")

    if visualization_choice == "Loss Over Epochs":
        st.image("loss_plot.png", caption="Loss Over Epochs", use_container_width=True)

    elif visualization_choice == "Accuracy Over Epochs":
        st.image("accuracy_plot.png", caption="Accuracy Over Epochs", use_container_width=True)

    elif visualization_choice == "Precision, Recall, F1-Score Over Epochs":
        st.image("prc_rec_f1_epoch.png", caption="Precision, Recall, and F1-Score Over Epochs", use_container_width=True)

    elif visualization_choice == "confusion_matrix":
        st.image("confusion_matrix.png", caption="confusion_matrix", use_container_width=True)

    else:
        st.warning("Select a visualization from the sidebar.")
    

    # Print more debugging information to console
    print(f"Predicted class index: {predicted_class_index}")
