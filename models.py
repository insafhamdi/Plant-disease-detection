#from flask_login import LoginManager
#from flask_login import UserMixin
from flask import session ,jsonify

 


#class User(UserMixin, db.Model):
"""
class User(UserMixin):
    
    def __init__ (self, user_json):
        self.user_json = user_json

    def get_id(self):
        object_id= self.user_json.get('_id')
        return str(object_id)
    def query(user_id):
        r=db1.users.find({ '_id': user_id})
        return r 
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
from PIL import Image
import time 
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.datasets as datasets
from torchmetrics import Accuracy, Precision, Recall, F1Score


class MultiClassCNN(nn.Module):
    def __init__(self):
        super(MultiClassCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.fc_block = nn.Sequential(
            nn.Linear(32 * 62 * 62, 128),  # Calculated based on the output size after convolutions
            nn.ReLU(),
            nn.Linear(128, 39)  # 39 classes
        ) 

    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x 

def learn1():

    model1 = MultiClassCNN()
    # Load the saved model
    saved_model_path =  'C:/Users/Babacar Gaye/saved_models/model_results.pth'
    checkpoint = torch.load(saved_model_path)

    # Load model state dict
    model1.load_state_dict(checkpoint['model_state_dict'])

    # Print additional information
    print("Accuracy:", checkpoint['accuracy'])
    print(" Precision:", checkpoint['precision'])
    print(" Recall:", checkpoint['recall'])
    print(" F1Score:", checkpoint['F1Score'])

   


    import torch.nn.functional as F
    dict_names_class={1: 'Apple___Apple_scab', 2: 'Apple___Black_rot', 3: 'Apple___Cedar_apple_rust', 4: 'Apple___healthy', 5: 'Background_without_leaves', 6: 'Blueberry___healthy', 7: 'Cherry___healthy', 8: 'Cherry___Powdery_mildew', 9: 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 10: 'Corn___Common_rust', 11: 'Corn___healthy', 12: 'Corn___Northern_Leaf_Blight', 13: 'Grape___Black_rot', 14: 'Grape___Esca_(Black_Measles)', 15: 'Grape___healthy', 16: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 17: 'Orange___Haunglongbing_(Citrus_greening)', 
    18: 'Peach___Bacterial_spot', 19: 'Peach___healthy', 20: 'Pepper,_bell___Bacterial_spot', 21: 'Pepper,_bell___healthy', 22: 'Potato___Early_blight', 23: 'Potato___healthy', 24: 'Potato___Late_blight', 25: 'Raspberry___healthy', 26: 'Soybean___healthy', 27: 'Squash___Powdery_mildew', 28: 'Strawberry___healthy', 29: 'Strawberry___Leaf_scorch', 30: 'Tomato___Bacterial_spot', 31: 'Tomato___Early_blight', 32: 'Tomato___healthy', 33: 'Tomato___Late_blight', 34: 'Tomato___Leaf_Mold', 35: 'Tomato___Septoria_leaf_spot', 36: 'Tomato___Spider_mites Two-spotted_spider_mite', 37: 'Tomato___Target_Spot', 38: 'Tomato___Tomato_mosaic_virus', 39: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'}
    # Define the path to your test images directory
    path = "C:/Users/Babacar Gaye/Desktop/mes docs babs/mes etudes/ESSAI 2EM ANNE/PFA/Test/Test_image"

    # Define the transformation to apply to each image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  
    ])

    # Function to load and preprocess an image
    def load_and_preprocess_image(image_path, transform):
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image

    # List all files in the directory
    image_files = os.listdir(path)

    #Iterate over each image file
    for image_file in image_files:
        image_path = os.path.join(path, image_file)
        
        # Load and preprocess the image
        image = load_and_preprocess_image(image_path, transform)
        
        # Pass the image through the model
        outputs = model1(image) 
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        predicted = torch.argmax(outputs, dim=1)
        
        # Get the probability associated with the predicted class
        confidence = probabilities[0][predicted.item()].item() * 100
        
        # Get the predicted class name from the dictionary
        predicted_class_name = dict_names_class[predicted.item()]
        
        # Print the predicted class name and confidence
        print(f"Image: {image_file}, Predicted Class: {predicted_class_name}, Confidence: {confidence:.2f}%")
        return {"Image": {image_file}, "Predicted Class": {predicted_class_name}, "Confidence": {confidence}} 


def learn():
    # Votre code existant...
    model1 = MultiClassCNN()
    # Load the saved model
    saved_model_path =  'C:/Users/Babacar Gaye/saved_models/model_results.pth'
    checkpoint = torch.load(saved_model_path)

    # Load model state dict
    model1.load_state_dict(checkpoint['model_state_dict'])

    # Print additional information
    print("Accuracy:", checkpoint['accuracy'])
    print(" Precision:", checkpoint['precision'])
    print(" Recall:", checkpoint['recall'])
    print(" F1Score:", checkpoint['F1Score'])

   


    import torch.nn.functional as F
    dict_names_class={1: 'Apple___Apple_scab', 2: 'Apple___Black_rot', 3: 'Apple___Cedar_apple_rust', 4: 'Apple___healthy', 5: 'Background_without_leaves', 6: 'Blueberry___healthy', 7: 'Cherry___healthy', 8: 'Cherry___Powdery_mildew', 9: 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 10: 'Corn___Common_rust', 11: 'Corn___healthy', 12: 'Corn___Northern_Leaf_Blight', 13: 'Grape___Black_rot', 14: 'Grape___Esca_(Black_Measles)', 15: 'Grape___healthy', 16: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 17: 'Orange___Haunglongbing_(Citrus_greening)', 
    18: 'Peach___Bacterial_spot', 19: 'Peach___healthy', 20: 'Pepper,_bell___Bacterial_spot', 21: 'Pepper,_bell___healthy', 22: 'Potato___Early_blight', 23: 'Potato___healthy', 24: 'Potato___Late_blight', 25: 'Raspberry___healthy', 26: 'Soybean___healthy', 27: 'Squash___Powdery_mildew', 28: 'Strawberry___healthy', 29: 'Strawberry___Leaf_scorch', 30: 'Tomato___Bacterial_spot', 31: 'Tomato___Early_blight', 32: 'Tomato___healthy', 33: 'Tomato___Late_blight', 34: 'Tomato___Leaf_Mold', 35: 'Tomato___Septoria_leaf_spot', 36: 'Tomato___Spider_mites Two-spotted_spider_mite', 37: 'Tomato___Target_Spot', 38: 'Tomato___Tomato_mosaic_virus', 39: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'}
    # Define the path to your test images directory
    path = "C:/Users/Babacar Gaye/Desktop/mes docs babs/mes etudes/ESSAI 2EM ANNE/PFA/Test/Test_image"

    # Define the transformation to apply to each image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  
    ])

    # Function to load and preprocess an image
    def load_and_preprocess_image(image_path, transform):
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image

    # List all files in the directory
    image_files = os.listdir(path)

    # Iterate over each image file
    predictions = []
    for image_file in image_files:
        image_path = os.path.join(path, image_file)
        
        # Load and preprocess the image
        image = load_and_preprocess_image(image_path, transform)
        
        # Pass the image through the model
        outputs = model1(image) 
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        predicted = torch.argmax(outputs, dim=1)
        
        # Get the probability associated with the predicted class
        confidence = probabilities[0][predicted.item()].item() * 100
        
        # Get the predicted class name from the dictionary
        predicted_class_name = dict_names_class[predicted.item()+1]
        
        # Add the prediction to the list
        predictions.append({
            "Image": image_file,
            "Predicted Class": predicted_class_name,
            "Confidence": confidence
        })

    # Return the predictions
    return predictions

def predict_single_image(image_path):
    model1 = MultiClassCNN()
    # Load the saved model
    def load_and_preprocess_image(image_path, transform):
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image

    saved_model_path = 'C:/Users/Babacar Gaye/saved_models/model_results.pth'
    checkpoint = torch.load(saved_model_path)

    # Load model state dict
    model1.load_state_dict(checkpoint['model_state_dict'])

    import torch.nn.functional as F
    dict_names_class={1: 'Apple___Apple_scab', 2: 'Apple___Black_rot', 3: 'Apple___Cedar_apple_rust', 4: 'Apple___healthy', 5: 'Background_without_leaves', 6: 'Blueberry___healthy', 7: 'Cherry___healthy', 8: 'Cherry___Powdery_mildew', 9: 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 10: 'Corn___Common_rust', 11: 'Corn___healthy', 12: 'Corn___Northern_Leaf_Blight', 13: 'Grape___Black_rot', 14: 'Grape___Esca_(Black_Measles)', 15: 'Grape___healthy', 16: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 17: 'Orange___Haunglongbing_(Citrus_greening)', 
    18: 'Peach___Bacterial_spot', 19: 'Peach___healthy', 20: 'Pepper,_bell___Bacterial_spot', 21: 'Pepper,_bell___healthy', 22: 'Potato___Early_blight', 23: 'Potato___healthy', 24: 'Potato___Late_blight', 25: 'Raspberry___healthy', 26: 'Soybean___healthy', 27: 'Squash___Powdery_mildew', 28: 'Strawberry___healthy', 29: 'Strawberry___Leaf_scorch', 30: 'Tomato___Bacterial_spot', 31: 'Tomato___Early_blight', 32: 'Tomato___healthy', 33: 'Tomato___Late_blight', 34: 'Tomato___Leaf_Mold', 35: 'Tomato___Septoria_leaf_spot', 36: 'Tomato___Spider_mites Two-spotted_spider_mite', 37: 'Tomato___Target_Spot', 38: 'Tomato___Tomato_mosaic_virus', 39: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus'}
    # Define the path to your test images directory

    # Define the transformation to apply to each image
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  
    ])

    # Load and preprocess the image
    image = load_and_preprocess_image(image_path, transform)
    
    # Pass the image through the model
    outputs = model1(image) 
    probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
    predicted = torch.argmax(outputs, dim=1)
    
    # Get the probability associated with the predicted class
    confidence = probabilities[0][predicted.item()].item() * 100
    
    # Get the predicted class name from the dictionary
    predicted_class_name = dict_names_class[predicted.item()]
    
    # Return the prediction as a dictionary
    return {
        "Image": os.path.basename(image_path),
        "Predicted Class": predicted_class_name,
        "Confidence": confidence
    }
