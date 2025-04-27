import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import CustomDataset
from model import CNN_BiGRU
from gesture import GESTURE

# Configuration
window_size = 30
batch_size = 64
epochs = 30
learning_rate = 0.0001
dataset_dir = './data/'
model_save_dir = './model/'
os.makedirs(model_save_dir, exist_ok=True)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset preparation
dataset = CustomDataset(window_size=window_size, folder_dir=dataset_dir)
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model initialization
model = CNN_BiGRU(input_size=99, output_size=64, units=32, num_classes=len(GESTURE)).to(device)

# Loss function & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, loader, device, optimizer, loss_func):
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

# Evaluation function
def evaluate(model, loader, device, loss_func):
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

# Training loop
for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}/{epochs}')
    train_loss, train_acc = train(model, train_loader, device, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, device, criterion)

    print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.6f}, Val Acc:   {val_acc:.2f}%")

# Save model
torch.save(model.state_dict(), os.path.join(model_save_dir, 'model_dict.pt'))
torch.save(model, os.path.join(model_save_dir, 'model.pt'))
print(f"\nModel saved successfully: {model_save_dir}")
