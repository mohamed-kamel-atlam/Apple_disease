# plant-disease-detection-project


⁉️ why this project?
---------------------
Make a deep learning model that can detect plant diseases (especially apple diseases) from images.
Early detection of plant diseases can help
1)farmers take timely actions, reducing crop losses and promoting sustainable agriculture.
enabling users to interactively diagnose plant diseases.

2)Agricultural Experts and Researchers: Agricultural experts and researchers can leverage this tool to identify plant diseases in experimental settings. The system can aid in analyzing disease prevalence and patterns across different plant species.

3)Educational Purposes: Students and educators can use this platform as an educational resource to understand plant diseases and learn about deep learning techniques applied in real-world applications.

4)Future Enhancements: The project's potential for growth is vast, and future enhancements could include:
Integration with additional plant-related datasets to improve the model's accuracy and generalization. Deployment on cloud platforms for scalability and increased availability. Incorporating real-time image augmentation and preprocessing techniques to handle various image qualities and formats. Building a mobile app version for on-the-go plant disease diagnosis.

🗂️ Dataset
--------------
 Source: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset.
 Content:87.9k images.
 this project use only apple images (7780 image).

 🛠️ Preparing data
 ------------------
 Used transforms.ToTensor() for normalization.
 initialize DataLoader for efficient batch processing.

🏋️Training
------------------
using:- 
CNN model : for image recognition and processing.
Adam optimizer: for per-parameter adaptive learning rates.
Learning Rate: that determines the step size at each iteration while moving toward a minimum of a loss function.
loss function: CrossEntropyLoss() to measure the performance of a classification (model Calculate loss).

📊 visualization
------------------
!["M:\Collage\Apple_disease\accuracy_plot.png"](https://github.com/user-attachments/assets/d2df6da1-a365-4952-a9fe-aec84d77b75d)

!["M:\Collage\Apple_disease\loss_plot.png"](https://github.com/user-attachments/assets/7d1220af-cfd5-4bc6-83d1-9780081862ad)

!["M:\Collage\Apple_disease\prc_rec_f1_epoch.png"](https://github.com/user-attachments/assets/6e885863-0078-4f34-adbe-5d16ee204bd8)

!["M:\Collage\Apple_disease\confusion_matrix.png"](https://github.com/user-attachments/assets/c835284d-2b35-4548-a6b7-8bc19e5c7c94)









🧪 Testing
-----------------
testing another images to predict diseases and get accuracy.

🏃‍♂️‍➡️ run project
------------------------
run filename.py .
Then  upload the image and make prediction.

📈 Results
---------------------
Accuracy: Achieved 97% accuracy on the validation dataset, Achieved 92.5% accuracy on the test dataset




