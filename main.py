import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model import AudioClassifier
from data.dataloader import GTZANDataset

# --- 3. Minimal Training Loop ---
def train():
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 
        'cuda' if torch.cuda.is_available()
        else 'cpu'
    )

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
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
