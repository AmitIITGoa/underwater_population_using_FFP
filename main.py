#!/usr/bin/env python3
"""
Single-file DeepFish pipeline: auto-download, train-if-needed, demo fish counting.

Usage: python main.py --data-dir ~/data/DeepFish --epochs 5 --demo-samples 6
"""

import os
import sys
import argparse
import random
import tarfile
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional

import requests
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm

# Constants
DEEPFISH_URL = "http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar"
REQUIREMENTS_TXT = """torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
pandas>=1.2.0
Pillow>=8.0.0
matplotlib>=3.3.0
tqdm>=4.60.0
requests>=2.25.0
"""

def ensure_requirements():
    """Write requirements.txt if it doesn't exist."""
    req_path = Path("requirements.txt")
    if not req_path.exists():
        print("📝 Creating requirements.txt...")
        req_path.write_text(REQUIREMENTS_TXT.strip())

def download_file(url: str, filepath: Path, chunk_size: int = 8192) -> bool:
    """Download file with progress bar."""
    try:
        print(f"📥 Downloading {url}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def extract_tar(tar_path: Path, extract_to: Path) -> bool:
    """Extract tar file with progress."""
    try:
        print(f"📦 Extracting {tar_path.name}...")
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(extract_to)
        return True
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def download_dataset(data_dir: Path) -> bool:
    """Download and extract DeepFish dataset if not exists."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    deepfish_dir = data_dir / "DeepFish"
    if deepfish_dir.exists() and list(deepfish_dir.glob("*")):
        print(f"✅ Dataset already exists at {deepfish_dir}")
        return True
    
    # Download
    tar_path = data_dir / "DeepFish.tar"
    if not tar_path.exists():
        if not download_file(DEEPFISH_URL, tar_path):
            return False
    
    # Extract
    if not extract_tar(tar_path, data_dir):
        return False
    
    # Verify extraction
    if not deepfish_dir.exists() or not list(deepfish_dir.glob("*")):
        print("❌ Extraction verification failed - dataset folder is empty")
        return False
    
    print(f"✅ Dataset ready at {deepfish_dir}")
    
    # Clean up tar file
    if tar_path.exists():
        tar_path.unlink()
    
    return True

class DeepFishDataset(Dataset):
    """DeepFish dataset for fish counting."""
    
    def __init__(self, data_dir: Path, split: str = "train", transform=None, img_size: int = 512):
        self.data_dir = Path(data_dir) / "DeepFish"
        self.split = split
        self.transform = transform
        self.img_size = img_size
        
        # Find image files
        self.image_paths = []
        self.labels = []
        
        # Look for images in various subdirectories
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_paths.extend(list(self.data_dir.rglob(ext)))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {self.data_dir}")
        
        # Try to load labels from CSV files or create dummy labels
        self.load_labels()
        
        print(f"📊 {split} dataset: {len(self.image_paths)} images")
    
    def load_labels(self):
        """Load or create labels for fish counting."""
        # Look for CSV files with labels
        csv_files = list(self.data_dir.rglob("*.csv"))
        
        if csv_files:
            # Try to find relevant CSV with count/label data
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'counts' in df.columns or 'count' in df.columns:
                        print(f"📋 Found labels in {csv_file.name}")
                        self.load_csv_labels(df)
                        return
                except Exception as e:
                    continue
        
        # Create dummy labels if no CSV found
        print("⚠️  No count labels found, creating dummy labels (0 for all images)")
        self.labels = [0] * len(self.image_paths)
    
    def load_csv_labels(self, df: pd.DataFrame):
        """Load labels from CSV dataframe."""
        # Map image names to counts
        count_col = 'counts' if 'counts' in df.columns else 'count'
        id_col = 'ID' if 'ID' in df.columns else df.columns[0]
        
        id_to_count = dict(zip(df[id_col], df[count_col]))
        
        self.labels = []
        for img_path in self.image_paths:
            # Try to match image name with ID
            img_name = img_path.stem
            count = id_to_count.get(img_name, 0)
            self.labels.append(float(count))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Could not load {img_path}: {e}")
            # Return a dummy image
            image = Image.new('RGB', (self.img_size, self.img_size), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return {
            'image': image,
            'count': label,
            'path': str(img_path)
        }

def create_transforms(img_size: int):
    """Create train and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class FishCounterCNN(nn.Module):
    """Lightweight CNN for fish counting."""
    
    def __init__(self, num_classes=1):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x.squeeze(-1)  # Remove last dimension for scalar output

def prepare_data(data_dir: Path, img_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and validation dataloaders."""
    train_transform, val_transform = create_transforms(img_size)
    
    # Create full dataset
    full_dataset = DeepFishDataset(data_dir, "train", train_transform, img_size)
    
    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update val dataset transform
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                epochs: int, lr: float, model_path: Path, device: torch.device):
    """Train the fish counting model."""
    print(f"🚀 Training on {device} for {epochs} epochs...")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in train_pbar:
            images = batch['image'].to(device)
            targets = batch['count'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                targets = batch['count'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, model_path)
            print(f"💾 Best model saved to {model_path}")
        
        scheduler.step(avg_val_loss)
    
    print(f"✅ Training completed! Best val loss: {best_val_loss:.4f}")

def run_demo(model: nn.Module, data_dir: Path, num_samples: int, runs_dir: Path, 
             img_size: int, device: torch.device):
    """Run demo on random images and save visualizations."""
    print(f"🎯 Running demo on {num_samples} samples...")
    
    # Create demo dataset
    _, val_transform = create_transforms(img_size)
    demo_dataset = DeepFishDataset(data_dir, "demo", val_transform, img_size)
    
    # Select random samples
    indices = random.sample(range(len(demo_dataset)), min(num_samples, len(demo_dataset)))
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = runs_dir / f"demo_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = demo_dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            true_count = sample['count'].item()
            img_path = sample['path']
            
            # Predict
            pred_count = model(image).item()
            
            # Load original image for visualization
            orig_img = Image.open(img_path).convert('RGB')
            
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original image
            ax1.imshow(orig_img)
            ax1.set_title(f"Original Image\nTrue Count: {true_count:.1f}")
            ax1.axis('off')
            
            # Predicted image
            ax2.imshow(orig_img)
            ax2.set_title(f"Predicted Count: {pred_count:.1f}\nError: {abs(pred_count - true_count):.1f}")
            ax2.axis('off')
            
            # Save
            output_path = output_dir / f"demo_{i+1}_{Path(img_path).stem}.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            results.append({
                'image': Path(img_path).name,
                'true_count': true_count,
                'pred_count': pred_count,
                'error': abs(pred_count - true_count)
            })
            
            print(f"  {i+1}/{len(indices)}: {Path(img_path).name} | True: {true_count:.1f} | Pred: {pred_count:.1f}")
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "results.csv", index=False)
    
    # Print summary
    mean_error = results_df['error'].mean()
    print(f"\n📊 Demo Results Summary:")
    print(f"   Mean Absolute Error: {mean_error:.2f}")
    print(f"   Results saved to: {output_dir}")
    
    return output_dir

def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="DeepFish counting pipeline")
    parser.add_argument('--data-dir', type=str, default="~/data/DeepFish", 
                      help="Dataset directory")
    parser.add_argument('--model-dir', type=str, default="~/models/deepfish_counter",
                      help="Model directory")
    parser.add_argument('--runs-dir', type=str, default="~/runs/deepfish_demo",
                      help="Demo outputs directory")
    parser.add_argument('--epochs', type=int, default=5, help="Training epochs")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--img-size', type=int, default=512, help="Input image size")
    parser.add_argument('--num-workers', type=int, default=2, help="DataLoader workers")
    parser.add_argument('--demo-samples', type=int, default=6, help="Demo sample count")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Expand paths
    data_dir = Path(args.data_dir).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    runs_dir = Path(args.runs_dir).expanduser()
    model_path = model_dir / "best.pt"
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Ensure requirements.txt exists
    ensure_requirements()
    
    # Step 1: Download dataset
    if not download_dataset(data_dir):
        print("❌ Failed to download dataset. Exiting.")
        sys.exit(1)
    
    # Step 2: Create model
    model = FishCounterCNN().to(device)
    
    # Step 3: Train or load model
    if model_path.exists():
        print(f"✅ Model found at {model_path} → skipping training")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("🏋️ Model not found → starting training")
        train_loader, val_loader = prepare_data(data_dir, args.img_size, args.batch_size, args.num_workers)
        train_model(model, train_loader, val_loader, args.epochs, args.lr, model_path, device)
    
    # Step 4: Run demo
    demo_dir = run_demo(model, data_dir, args.demo_samples, runs_dir, args.img_size, device)
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"   Model: {model_path}")
    print(f"   Demo: {demo_dir}")

if __name__ == "__main__":
    main()
