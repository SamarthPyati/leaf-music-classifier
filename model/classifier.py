import torch.nn as nn
import torch.nn.functional as F

import os 
import sys 

# Ensure leaf-pytorch is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..' , 'leaf-pytorch'))

from leaf_pytorch.frontend import Leaf  # type: ignore

# import os 
# import sys
# sys.path.append(os.path.join(os.getcwd(), '..'))

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=10, sample_rate=16000, n_filters=40):
        super(AudioClassifier, self).__init__()
        
        # 1. Learnable Audio Frontend
        self.leaf = Leaf(
            sample_rate=sample_rate, 
            n_filters=n_filters,
            init_min_freq=60.0,
            init_max_freq=7800.0
        )
        
        # 2. 2D CNN Backbone
        # Input shape from LEAF: (Batch, n_filters, Time)
        # We will unsqueeze it to: (Batch, 1, n_filters, Time) to treat it as a 2D image
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.3)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Pools (Freq, Time) down to 1x1
            nn.Dropout(0.4)
        )
        
        # 3. Classifier Head
        self.fc1 = nn.Linear(256, 128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1. LEAF
        x = self.leaf(x)  # Shape: (B, 40, Time)
        
        # 2. Add channel dimension for 2D Convolutions
        x = x.unsqueeze(1) # Shape: (B, 1, 40, Time)
        
        # 3. 2D Convolutions
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # 4. Flatten
        x = x.view(x.size(0), -1) # Shape: (B, 256)
        
        # 5. Fully Connected
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        
        return x