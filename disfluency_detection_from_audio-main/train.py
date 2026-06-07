"""
Model Training Script
For training or fine-tuning the stuttering detection models
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from models import AcousticModel, MultimodalModel

class AudioDataset(Dataset):
    """Custom dataset for audio and labels"""
    
    def __init__(self, audio_dir, labels_dir, sr=16000, duration=5):
        """
        Initialize dataset
        Args:
            audio_dir: Directory containing audio files
            labels_dir: Directory containing label files
            sr: Sample rate
            duration: Audio duration in seconds
        """
        self.audio_dir = Path(audio_dir)
        self.labels_dir = Path(labels_dir)
        self.sr = sr
        self.samples_per_audio = sr * duration
        
        # Get list of audio files
        self.audio_files = list(self.audio_dir.glob('*.wav'))
        logger.info(f"Found {len(self.audio_files)} audio files")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """Get audio and corresponding labels"""
        try:
            import torchaudio
            
            audio_path = self.audio_files[idx]
            label_path = self.labels_dir / f"{audio_path.stem}.csv"
            
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(sr, self.sr)
                audio = resampler(audio)
            
            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)
            
            # Pad or truncate
            if audio.shape[1] < self.samples_per_audio:
                audio = torch.nn.functional.pad(audio, (0, self.samples_per_audio - audio.shape[1]))
            else:
                audio = audio[:, :self.samples_per_audio]
            
            # Load labels
            labels = torch.zeros(5, audio.shape[1] // self.sr)  # 5 disfluency types
            if label_path.exists():
                df = pd.read_csv(label_path)
                # Process labels (implementation depends on format)
                pass
            
            return audio.squeeze(0), labels
        
        except Exception as e:
            logger.error(f"Error loading {self.audio_files[idx]}: {str(e)}")
            # Return zero tensors on error
            return torch.zeros(self.samples_per_audio), torch.zeros(5)

class Trainer:
    """Trainer class for model training"""
    
    def __init__(self, model, device, learning_rate=1e-4, batch_size=16):
        """Initialize trainer"""
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.batch_size = batch_size
        self.best_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (audio, labels) in enumerate(train_loader):
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(self.model, AcousticModel):
                _, logits = self.model(audio.unsqueeze(1))
            else:
                logits = self.model(audio, audio)  # Placeholder for multimodal
            
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for audio, labels in val_loader:
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                
                if isinstance(self.model, AcousticModel):
                    _, logits = self.model(audio.unsqueeze(1))
                else:
                    logits = self.model(audio, audio)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=10, save_path='models/trained_model.pt'):
        """Complete training loop"""
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            logger.info(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate(val_loader)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"✓ Model saved with loss: {val_loss:.4f}")

def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description='Train disfluency detection models')
    parser.add_argument('--model', type=str, choices=['acoustic', 'multimodal'], 
                      default='acoustic', help='Model to train')
    parser.add_argument('--audio-dir', type=str, required=True, help='Directory with training audio')
    parser.add_argument('--labels-dir', type=str, required=True, help='Directory with labels')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-path', type=str, default='demo_models/trained_model.pt',
                      help='Path to save trained model')
    parser.add_argument('--pretrained', type=str, help='Path to pretrained model to fine-tune')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Stuttering Disfluency Detection - Model Training")
    logger.info("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create dataset
    logger.info("\nLoading dataset...")
    dataset = AudioDataset(args.audio_dir, args.labels_dir)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Training set: {len(train_set)} samples")
    logger.info(f"Validation set: {len(val_set)} samples")
    
    # Initialize model
    logger.info(f"\nInitializing {args.model} model...")
    if args.model == 'acoustic':
        model = AcousticModel()
    else:
        model = MultimodalModel()
    
    # Load pretrained weights if provided
    if args.pretrained:
        logger.info(f"Loading pretrained weights from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
    
    # Initialize trainer
    trainer = Trainer(model, device, args.learning_rate, args.batch_size)
    
    # Train model
    logger.info("\nStarting training...")
    trainer.train(train_loader, val_loader, args.epochs, args.save_path)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Training complete!")
    logger.info(f"✓ Model saved to: {args.save_path}")
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
