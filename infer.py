import argparse
import json
import torch
from torchvision import transforms
from PIL import Image
from model import FruitClassifier

def load_class_mapping(json_path: str):
    # Load class_to_idx and build inverse
    with open(json_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    return idx_to_class

def load_model(checkpoint_path: str, device: torch.device, num_classes: int):
    model = FruitClassifier(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    return model

def infer_image(image_path: str, model: torch.nn.Module, idx_to_class: dict, device: torch.device):
    # 1) Preprocess exactly as during validation
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    # 2) Forward
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)
        pred_idx = pred_idx.item()
        conf = conf.item()

    # 3) Map back to class name
    pred_class = idx_to_class[pred_idx]
    print(f"Predicted: {pred_class}  (index={pred_idx}, confidence={conf*100:.1f}%)")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("image_path", help="Path to input image")
    p.add_argument("--checkpoint", required=True, help="model .pth checkpoint")
    p.add_argument("--class_map", default="class_to_idx.json", help="class_to_idx JSON")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    idx_to_class = load_class_mapping(args.class_map)
    model = load_model(args.checkpoint, device, num_classes=len(idx_to_class))
    infer_image(args.image_path, model, idx_to_class, device)

if __name__ == "__main__":
    main()
