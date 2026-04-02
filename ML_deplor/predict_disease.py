import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from PIL import Image
import io
import os

# Define the standard 38 PlantVillage classes
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Define Model Architecture exactly as trained
class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.model = efficientnet_b0(pretrained=False)
        self.model.classifier = nn.Identity()
        
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.model(x) 
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize and load model globally so it stays in memory
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'plant_disease', 'best_plant_model.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plant_model = PlantDiseaseModel(num_classes=len(CLASS_NAMES))
try:
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True) # use weights_only=False if this fails, but usually better
    plant_model.load_state_dict(state_dict)
    plant_model.to(device)
    plant_model.eval()
    print("Plant Disease Model loaded successfully.")
except Exception as e:
    print(f"Warning: Failed to load Plant Disease Model from {MODEL_PATH}. Error: {e}")
    # Fallback to try weights_only=False which is sometimes needed for older save formats
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        plant_model.load_state_dict(state_dict)
        plant_model.to(device)
        plant_model.eval()
        print("Plant Disease Model loaded successfully (weights_only=False).")
    except Exception as e2:
         print(f"Error loading Plant Disease Model: {e2}")

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_plant_disease(image_bytes):
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess
        input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension
        
        # Inference
        with torch.no_grad():
            output = plant_model(input_tensor)
            _, predicted_idx = torch.max(output, 1)
            
        class_idx = predicted_idx.item()
        predicted_class = CLASS_NAMES[class_idx]
        
        # Format the class name to be more readable
        formatted_class = predicted_class.replace('___', ' - ').replace('_', ' ')
        
        return {
            "disease": formatted_class,
            "raw_class": predicted_class
        }
    except Exception as e:
        raise Exception(f"Image processing failed: {str(e)}")
