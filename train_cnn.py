import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MNISTDataset(Dataset):
    def __init__(self, csv_file, train=True):
        self.data = pd.read_csv(csv_file)
        self.train = train
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.train:
            label = self.data.iloc[idx, 0]
            pixels = self.data.iloc[idx, 1:].values.astype(np.float32) / 255.0
            pixels = pixels.reshape(1, 28, 28)
            return pixels, label
        else:
            pixels = self.data.iloc[idx, :].values.astype(np.float32) / 255.0
            pixels = pixels.reshape(1, 28, 28)
            return pixels

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = running_train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        train_losses.append(train_loss)
        
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = running_val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return train_losses, val_losses

def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    print('Saved loss_curve.png')

def predict(model, test_loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
    
    return predictions

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    full_dataset = MNISTDataset('train.csv', train=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    test_dataset = MNISTDataset('test.csv', train=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = SimpleCNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print('Starting training...')
    train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
    
    print('Plotting loss curve...')
    plot_loss_curve(train_losses, val_losses)
    
    print('Making predictions...')
    predictions = predict(model, test_loader, device)
    
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions)+1),
        'Label': predictions
    })
    submission.to_csv('sample_submission.csv', index=False)
    print('Saved sample_submission.csv')
    
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    print('Saved model to mnist_cnn.pth')

if __name__ == '__main__':
    main()