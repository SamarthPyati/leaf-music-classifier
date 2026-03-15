# Music Genre Classifier

## Description
A Deep Learning pipeline built with PyTorch and torchaudio to classify audio files into 10 distinct musical genres. This project leverages Google's LEAF (Learnable Audio Frontend) combined with a robust 2D Convolutional Neural Network (CNN) backbone to extract and learn complex features directly from raw audio waveforms.

## Project Structure
- `main.py`: Contains the `GTZANDataset` class for loading and preprocessing audio data, and the `AudioClassifier` model architecture (LEAF + 2D CNN).
- `notebooks/leaf_main_training.ipynb`: A Jupyter Notebook providing the complete training pipeline, including dataset extraction, DataLoader setup, the training loop with Automatic Mixed Precision (AMP), weight decay, and a learning rate scheduler to prevent overfitting.
- `inference.py`: A script for running predictions on single audio files using the trained model. It handles audio chunking and averaging to provide robust classification.

## Setup and Installation
This project uses `uv` for fast dependency management and environment isolation.

1. **Install `uv` (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or via Homebrew on macOS:
   ```bash
   brew install uv
   ```

2. **Clone the repository and set up the environment:**
   ```bash
   git clone <your-repo-url>
   cd music-classifier
   
   # Let uv install packages defined in pyproject.toml and sync the lockfile
   uv sync
   ```

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

4. **Install remaining ML dependencies required for this project:**
   ```bash
   uv pip install torch torchaudio scikit-learn soundfile
   ```

5. **Ensure `leaf-pytorch` is present:**
   The codebase relies on Google's LEAF. Ensure the `leaf-pytorch` directory is either cloned as a submodule or downloaded into the root directory of this project (`music-classifier/leaf-pytorch`).

## Dataset
The model is trained on the GTZAN Genre Collection dataset. 

If using the Kaggle `archive.zip` export, the `notebooks/leaf_main_training.ipynb` notebook handles the extraction and structures the paths correctly. 

The default expected structure after extraction is:
`Data/genres_original/<genre>/<file.wav>`

## Usage

### Training
The training process is intended to be run in an environment with GPU acceleration (like Google Colab).
1. Ensure the `leaf-pytorch` directory is present in the project root.
2. Upload the GTZAN `archive.zip` to your Jupyter environment.
3. Open `notebooks/leaf_main_training.ipynb`.
4. Run all cells sequentially. The notebook will automatically extract `archive.zip`, setup the data loaders, instantiate the model, and commence training. The trained weights will be saved as `music_genre_classifier.pth` in the notebook environment.

### Inference
To predict the genre of a new audio file locally, activate your `uv` environment and use the `inference.py` script. The script automatically uses CUDA or MPS if available.

```bash
python inference.py <path_to_audio_file> --model <path_to_model_weights.pth>
```

**Example:**
```bash
python inference.py data/mock/test_track.wav --model music_genre_classifier.pth
```

#### Testing the Model
To run predictions on a sample of test files from each genre in the GTZAN dataset:

```bash
python inference.py --test --model <path_to_model_weights.pth>
```

This will classify one file from each of the 10 genres and display the results.