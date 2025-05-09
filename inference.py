import json
import torch
from torchvision import transforms
from PIL import Image
from model import FruitClassifier

class InferenceModel:
    _instance = None

    def __new__(cls, checkpoint_path, class_map_path, device):
        if cls._instance is None:
            # load once
            cls._instance = super().__new__(cls)
            cls._instance.device = device
            # load class map
            with open(class_map_path) as f:
                class_to_idx = json.load(f)
            cls._instance.idx_to_class = {int(v):k for k,v in class_to_idx.items()}
            # load model
            model = FruitClassifier(num_classes=len(cls._instance.idx_to_class), pretrained=False)
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device).eval()
            cls._instance.model = model
            # transform pipeline
            cls._instance.transform = transforms.Compose([
                transforms.Resize((100,100)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        return cls._instance

    def predict(self, pil_image):
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor), dim=1)
            conf, idx = probs.max(1)
        return self.idx_to_class[idx.item()], conf.item()
