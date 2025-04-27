# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import CustomDataset
from model import CNN_BiGRU
from config import GESTURE

class TrainingConfig:
    def __init__(self):
        self.window_size = 30
        self.batch_size = 64
        self.epochs = 30
        self.learning_rate = 0.0001
        self.dataset_dir = './data/'
        self.model_save_dir = './model/'
        self.val_ratio = 0.2

        # Create model directory
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_dataloaders(config):
    """Prepare train and validation dataloaders"""
    dataset = CustomDataset(window_size=config.window_size, folder_dir=config.dataset_dir)

    val_size = int(len(dataset) * config.val_ratio)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader

def train_epoch(model, loader, device, optimizer, loss_func):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0

    for x_batch, y_batch in tqdm(loader):
        y_batch = y_batch.type(torch.LongTensor).to(device).squeeze()
        x_batch = x_batch.to(device)

        pred = model(x_batch)
        loss = loss_func(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (pred.argmax(dim=1) == y_batch).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    return avg_loss, accuracy

def evaluate(model, loader, device, loss_func):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            y_batch = y_batch.type(torch.LongTensor).to(device).squeeze()
            x_batch = x_batch.to(device)

            pred = model(x_batch)
            loss = loss_func(pred, y_batch)

            total_loss += loss.item()
            correct += (pred.argmax(dim=1) == y_batch).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    return avg_loss, accuracy

def save_model(model, save_dir):
    """Save model weights and architecture"""
    torch.save(model.state_dict(), os.path.join(save_dir, 'model_dict.pt'))
    torch.save(model, os.path.join(save_dir, 'model.pt'))
    print(f"\nModel saved successfully: {save_dir}")

def train_model(config):
    """Main training function"""
    # Prepare data
    train_loader, val_loader = prepare_dataloaders(config)

    # Initialize model
    model = CNN_BiGRU(
        input_size=99,
        output_size=64,
        units=32,
        num_classes=len(GESTURE)
    ).to(config.device)

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(config.epochs):
        print(f'\nEpoch {epoch + 1}/{config.epochs}')

        train_loss, train_acc = train_epoch(
            model, train_loader, config.device, optimizer, criterion
        )
        val_loss, val_acc = evaluate(
            model, val_loader, config.device, criterion
        )

        print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.6f}, Val Acc:   {val_acc:.2f}%")

    # Save the trained model
    save_model(model, config.model_save_dir)

def main():
    # Initialize configuration
    config = TrainingConfig()

    # Start training
    train_model(config)

if __name__ == "__main__":
    main()
