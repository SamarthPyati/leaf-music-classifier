import argparse
import torch
import torchaudio
import torch.nn.functional as F
import os
from pathlib import Path 
import sys

# Ensure leaf-pytorch is in path if not already
sys.path.append(os.path.join(os.path.dirname(__file__), 'leaf-pytorch'))
from main import AudioClassifier, GTZANDataset

def predict(audio_path, model_path):
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )

    print(f"Using device: {device}")
    
    # Initialize model
    model = AudioClassifier(num_classes=10).to(device)
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return
        
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Load audio
    target_sample_rate = 16000
    chunk_duration = 5.0
    chunk_samples = int(target_sample_rate * chunk_duration)
    
    try:
        waveform, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return
        
    # Resample if needed
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)
        
    # Ensure mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    # Split into 5-second chunks
    total_samples = waveform.shape[1]
    chunks = []
    
    for start_idx in range(0, total_samples, chunk_samples):
        end_idx = start_idx + chunk_samples
        chunk = waveform[:, start_idx:end_idx]
        
        # Pad the last chunk if it's too short
        if chunk.shape[1] < chunk_samples:
            pad_amount = chunk_samples - chunk.shape[1]
            chunk = F.pad(chunk, (0, pad_amount))
            
        chunks.append(chunk)

    if not chunks:
        print("Audio file is empty.")
        return

    # Add batch dimension and move to device
    # Shape: (Num_Chunks, 1, 80000)
    batch_waveform = torch.stack(chunks).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(batch_waveform)
        # Average the probabilities across all chunks
        probabilities = F.softmax(outputs, dim=1).mean(dim=0)
        predicted = probabilities.argmax()
        
    genres = [
        "blues", "classical", "country", "disco", "hiphop",
        "jazz", "metal", "pop", "reggae", "rock"
    ]
    
    predicted_genre = genres[predicted.item()]
    confidence = probabilities[predicted.item()].item() * 100
    
    print(f"\nPredicted Genre: {predicted_genre} (Confidence: {confidence:.2f}%)")

def fetch_test_files(): 
    dataset = GTZANDataset()
    files = [dataset.__dict__['files'][(i + 1)*100 - 1] for i in range(10)]
    return files

def test_run(model_path):
    if not Path(model_path).exists():
        print("Model does not exists at path: ", model_path)
        return 

    files = fetch_test_files()
    for audio_file, genre in files:
        print("-" * 50)
        print(f"Audio File: {Path(audio_file).name}, Genre: {genre}")
        predict(audio_file, model_path)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Music Genre Inference")
    parser.add_argument("--model", default="music_genre_classifier.pth", help="Path to the trained PyTorch model")
    parser.add_argument("--test", action="store_true", help="Make a prediction on all genre files from the GTZAN dataset.")
    parser.add_argument("audio_path", nargs="?", help="Path to the audio file (.wav, .au, etc.)")
    
    args = parser.parse_args()
    if args.test: 
        test_run(args.model)
    else: 
        if not args.audio_path:
            parser.error("audio_path is required when --test is not used")
        predict(args.audio_path, args.model)
