import argparse
from pathlib import Path
import random
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
from tqdm import tqdm 

from dataset import FruitsDataset
from model import FruitClassifier


def freeze_all_backbone(model : FruitClassifier):
    for param in model.backbone.parameters():
            param.requires_grad = False
            
    for name, param in model.named_parameters():
        print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")

def unfreeze_l4_backbone(model: FruitClassifier):
    for name, param in model.named_parameters():
        if name.startswith("backbone.7"):
            param.requires_grad = True
            
    for name, param in model.named_parameters():
        print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")
        
def unfreeze_all_backbone(model: FruitClassifier):
    for param in model.backbone.parameters():
        param.requires_grad = True
        
    for name, param in model.named_parameters():
        print(f"{name}: {'trainable' if param.requires_grad else 'frozen'}")

def train_one_epoch(model : FruitClassifier, criterion : nn.CrossEntropyLoss, optimizer : optim.Optimizer, train_loader : DataLoader, device : torch.device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(train_loader, desc="Training", leave=False):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(images)
        _, predicted = torch.max(predictions, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

        loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update statistics
        running_loss += loss.item()

    
    # Compute average losses
    num_batches = len(train_loader)
    avg_loss = running_loss / num_batches
    avg_acc = correct / total
    
    return avg_loss, avg_acc    

def validate(model : FruitClassifier, criterion : nn.CrossEntropyLoss, val_loader : DataLoader, device : torch.device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="Validation", leave=False):
            images = images.to(device)
            targets = targets.to(device)
        
            predictions = model(images)
            _, predicted = torch.max(predictions, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            loss = criterion(predictions, targets)
            running_loss += loss.item()
            
    num_batches = len(val_loader)   
    avg_loss = running_loss / num_batches
    avg_acc = correct / total
    
    return avg_loss, avg_acc
    


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze_backbone", type=bool, default=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    workers = args.workers
    device = args.device
    save_dir = args.save_dir
    resume = args.resume
    freeze_backbone = args.freeze_backbone
    
    device = torch.device(device)
    Path(args.save_dir).mkdir(exist_ok=True)


    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dataset = FruitsDataset(transform=train_transform, download=True)
    val_dataset = FruitsDataset(transform=test_transform, split="val")
    test_dataset = FruitsDataset(transform=test_transform, split="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
    
    writer = SummaryWriter(log_dir=save_dir + '/tb_logs')

    
    model = FruitClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, milestones=[15,35], gamma=0.1)

    
    freeze_all_backbone(model)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0
    if resume:
        checkpoint = torch.load(
            resume,
            map_location=device,
            weights_only=False
        )        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'] )  

        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resumed from {resume}: starting at epoch {start_epoch}, best_acc={best_acc:.4f}")
        
    # Training loop
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        
        if epoch == 15:
            unfreeze_l4_backbone(model)
        if epoch == 35:
            unfreeze_all_backbone(model) 
        
        # Train
        train_loss, train_acc = train_one_epoch(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader, device=device)
        
        # Validate
        val_loss, val_acc = validate(model=model, criterion=criterion, val_loader=val_loader, device=device)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Acc/train', train_acc,   epoch)
        writer.add_scalar('Loss/val',   val_loss,   epoch)
        writer.add_scalar('Acc/val',     val_acc,    epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)
        
        if epoch == 0:
            dummy = torch.zeros((1, 3, 100, 100), device=device)
            writer.add_graph(model, dummy)  # :contentReference[oaicite:1]{index=1}


        
        # Print progress
        print(f"Epoch {epoch:>2} | time: {(time.time()-t0):.1f}s"
              f" | Train Loss: {train_loss:.4f}"
              f" | Train Acc: {train_acc:.4f}"
              f" | Val Loss: {val_loss:.4f}"
              f" | Val Acc: {val_acc:.4f}")
        
        # Save checkpoint if best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc
            }
            path = Path(save_dir) / f"best_acc_{best_acc:.4f}_ep{epoch}.pth"
            torch.save(checkpoint, path)
            print(f"→ Saved best checkpoint: {path}")
        
            scheduler.step()
    
    writer.flush()
    writer.close()


    print("\nEvaluating on test set...")
    test_loss, test_acc = validate(model=model, criterion=criterion, val_loader=test_loader, device=device)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    print("Training complete.")
    
if __name__ == "__main__":
    main()
        
        
    

