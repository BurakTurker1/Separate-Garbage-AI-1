import os
import torch
from torchvision.transforms import ToTensor, Resize, Compose
from torchvision.models import resnet50
# from torchvision import transforms
import torch.nn as nn
from PIL import Image

# Load the trained ResNet-50 model
model = resnet50(pretrained=False)  # Assume the model is not pretrained in this case
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)  # Change to the number of classes in your case

# Load the trained weights
model_path = 'resnet50_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
else:
    print(f"{model_path} not found. Please check the file path.")

# Transformation for input image
transform = Compose([Resize((256, 256)), ToTensor()])


def predict_image_category(image_path):
    # Open and preprocess the image
    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    # Print the predicted category
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # Change to your class names
    predicted_class = class_names[predicted.item()]
    print(f'The predicted class is: {predicted_class}')

# Example usage


image_path = 'clearlyPhoto.png'  # Replace with the path to your image
if os.path.exists(image_path):
    predict_image_category(image_path)
else:
    print(f"{image_path} not found. Please check the file path.")
