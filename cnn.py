import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=62):
        super(SimpleCNN, self).__init__()
        # Input: (1, 64, 64)
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) # Output: (32, 32, 32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Output: (64, 16, 16)
        
        # 64 channels * 16 * 16 spatial dim = 16384
        self.fc1 = nn.Linear(64 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1) 
        
        # Classifier
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class DeepLearningOCR:
    def __init__(self, num_classes=62, epochs=20, batch_size=64, learning_rate=0.001):
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            
        print(f"Initializing CNN on device: {self.device}")
        
        self.model = SimpleCNN(num_classes=num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X, y):
        # Convert lists to a single Batch Tensor
        # stack expects a list of tensors, so we ensure they are tensors first
        if isinstance(X, list):
            X_tensor = torch.stack(X) 
        else:
            X_tensor = torch.tensor(X)
            
        y_tensor = torch.tensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Starting training for {self.epochs} epochs...")
        self.model.train() # enables Dropout
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            start_t = time.time()
            
            for i, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{self.epochs}] Loss: {running_loss/len(loader):.4f} - Time: {time.time()-start_t:.1f}s")

    def predict(self, X):
        self.model.eval() # disables Dropout
        
        if isinstance(X, list):
            X_tensor = torch.stack(X)
        else:
            X_tensor = torch.tensor(X)
            
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_preds = []
        
        with torch.no_grad():
            for (inputs,) in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                
        return np.array(all_preds)