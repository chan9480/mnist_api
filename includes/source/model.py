import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 28x28 -> 28x28
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Reduce spatial dimensions by half

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Flattened size from 3x3 feature map
        self.fc2 = nn.Linear(512, 10)  # Output layer for 10 classes (MNIST digits)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # Second convolutional block
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Third convolutional block
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten the output for fully connected layers
        x = x.view(-1, 128 * 3 * 3)  # Adjusted for new feature map size
        
        # First fully connected layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout for regularization

        # Final output layer
        x = self.fc2(x)
        return x

