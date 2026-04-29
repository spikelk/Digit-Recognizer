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

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=5):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
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
        train_accs.append(train_acc)
        
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
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print('Early stopping triggered')
            break
    
    return train_losses, val_losses, train_accs, val_accs, epoch+1

def plot_comparison(results):
    plt.figure(figsize=(14, 8))
    
    plt.subplot(1, 2, 1)
    for optimizer_name, data in results.items():
        plt.plot(range(1, len(data['val_losses'])+1), data['val_losses'], label=f'{optimizer_name} - Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    for optimizer_name, data in results.items():
        plt.plot(range(1, len(data['val_accs'])+1), data['val_accs'], label=f'{optimizer_name} - Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png')
    print('Saved optimizer_comparison.png')

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
    
    optimizers = {
        'Adam': optim.Adam,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop,
        'Adagrad': optim.Adagrad,
        'AdamW': optim.AdamW
    }
    
    results = {}
    best_optimizer = None
    best_val_acc = 0
    best_model_state = None
    
    for optimizer_name, optimizer_class in optimizers.items():
        print(f'\n{"="*60}')
        print(f'Training with {optimizer_name} optimizer')
        print(f'{"="*60}')
        
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        
        if optimizer_name == 'SGD':
            optimizer = optimizer_class(model.parameters(), lr=0.01, momentum=0.9)
        else:
            optimizer = optimizer_class(model.parameters(), lr=0.001)
        
        train_losses, val_losses, train_accs, val_accs, epochs_trained = train_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer, device, epochs=50, patience=5
        )
        
        results[optimizer_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'final_val_acc': val_accs[-1],
            'epochs_trained': epochs_trained
        }
        
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            best_optimizer = optimizer_name
            best_model_state = model.state_dict().copy()
    
    print(f'\n{"="*60}')
    print('Training Results Summary')
    print(f'{"="*60}')
    
    for optimizer_name, data in results.items():
        print(f'{optimizer_name}: Final Val Acc = {data["final_val_acc"]:.2f}%, Epochs = {data["epochs_trained"]}')
    
    print(f'\nBest optimizer: {best_optimizer} with Val Acc = {best_val_acc:.2f}%')
    
    print('\nPlotting comparison...')
    plot_comparison(results)
    
    print('\nSaving best model...')
    torch.save(best_model_state, 'mnist_cnn.pth')
    print(f'Saved best model (trained with {best_optimizer}) to mnist_cnn.pth')
    
    print('\nMaking predictions with best model...')
    best_model = SimpleCNN().to(device)
    best_model.load_state_dict(best_model_state)
    predictions = predict(best_model, test_loader, device)
    
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions)+1),
        'Label': predictions
    })
    submission.to_csv('sample_submission.csv', index=False)
    print('Saved sample_submission.csv')

if __name__ == '__main__':
    main()