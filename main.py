import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import soundfile as sf
import os
import sys

# Ensure leaf-pytorch is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'leaf-pytorch'))
from leaf_pytorch.frontend import Leaf

# --- 1. Dataset ---
import torchaudio

class GTZANDataset(Dataset):
    def __init__(self, root="data/gtzan/genres_original", sample_rate=16000, duration=5.0):
        self.root = root
        self.target_sample_rate = sample_rate
        self.max_length = int(sample_rate * duration)
        
        self.genres = [
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "metal", "pop", "reggae", "rock"
        ]
        self.genre_to_idx = {g: i for i, g in enumerate(self.genres)}
        
        self.files = []
        for genre in self.genres:
            genre_dir = os.path.join(root, genre)
            if not os.path.exists(genre_dir):
                continue
            for f in os.listdir(genre_dir):
                if f.endswith('.wav'):
                    self.files.append((os.path.join(genre_dir, f), genre))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception:
            # Some GTZAN files are known to be corrupted (e.g., jazz.00054.wav)
            # Return a zero tensor if parsing fails
            waveform = torch.zeros(1, self.max_length)
            sr = self.target_sample_rate
        
        # Resample if needed
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
            
        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or truncate
        if waveform.shape[1] > self.max_length:
            waveform = waveform[:, :self.max_length]
        elif waveform.shape[1] < self.max_length:
            pad_amount = self.max_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_amount))
            
        label_idx = self.genre_to_idx[label]
        return waveform, label_idx


# --- 2. Model Pipeline ---
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

# --- 3. Minimal Training Loop ---
def train():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup data
    print("Setting up GTZAN dataset from local files...")
    full_dataset = GTZANDataset()
    
    if len(full_dataset) == 0:
        print("Error: No files found. Ensure dataset is extracted to data/gtzan/genres_original/")
        return
        
    print(f"Total dataset size: {len(full_dataset)}")
    
    # Train / Val Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Initialize model
    print("Initializing model for 10 genre classes...")
    model = AudioClassifier(num_classes=10).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    epochs = 10
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation Loop
        model.eval()
        val_loss = 0
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
        val_loss = val_loss / len(val_loader)
        scheduler.step(val_loss)
        
        # Calculate Scikit-Learn Metrics (macro to aggregate correctly across 10 classes)
        val_acc = accuracy_score(all_targets, all_preds) * 100.
        val_prec = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        val_rec = recall_score(all_targets, all_preds, average='macro', zero_division=0)
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"             | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, F1: {val_f1:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f}")
        
    print("\nTraining complete! Pipeline verified successfully.")

if __name__ == '__main__':
    train()
