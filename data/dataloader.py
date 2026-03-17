import os 
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

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