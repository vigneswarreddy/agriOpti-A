import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.model = efficientnet_b0(pretrained=False)
        self.model.classifier = nn.Identity()
        
        self.fc1 = nn.Linear(1280, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.model(x) # efficientnet_b0 with Identity classifier outputs (batch, 1280)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    model = PlantDiseaseModel()
    model_path = r'C:\agriopti\AgriOpti\AgriOpti\AgriOpti\ML\models\plant_disease\best_plant_model.pth'
    
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Let's count keys
        expected_keys = set(model.state_dict().keys())
        actual_keys = set(state_dict.keys())
        
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
        
        if len(missing) == 0 and len(unexpected) == 0:
            print("\nArchitecture match SUCCESS! The state_dict can be loaded perfectly.")
            model.load_state_dict(state_dict)
    except Exception as e:
        print("Error:", e)
