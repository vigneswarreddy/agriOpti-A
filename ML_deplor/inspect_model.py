import torch

model_path = r'C:\agriopti\AgriOpti\AgriOpti\AgriOpti\ML\models\plant_disease\best_plant_model.pth'

try:
    data = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    with open('model_layers.txt', 'w') as f:
        if isinstance(data, dict):
            for k, v in data.items():
                if hasattr(v, 'shape'):
                    f.write(f"{k}: {v.shape}\n")
        
except Exception as e:
    print("Error loading model:", e)
