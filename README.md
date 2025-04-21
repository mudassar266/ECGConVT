ECGConVT/
├── data/
│   ├── raw/           # Original ECG image folders (MI, PMI, AHB, Normal)
│   └── processed/     # Preprocessed .npy, TFRecord, or resized images
├── src/               # Source code modules
│   ├── __init__.py
│   ├── config.py      # YAML config loader
│   ├── data_loader.py # Dataset loading & preprocessing
│   ├── xception_model.py
│   ├── vit_model.py
│   ├── fusion_model.py
│   ├── train.py       # Training script
│   ├── evaluate.py    # Evaluation & metrics
│   └── utils.py       # Plotting & helpers
├── notebooks/         # Jupyter notebooks for exploration & reports
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_architectures.ipynb
│   └── 03_results_analysis.ipynb
├── scripts/           # Shell scripts (run_train.sh, run_eval.sh)
├── reports/           # Saved figures & metrics (classification_report.json, cm_*.png)
├── config.yaml        # Experiment configuration
├── environment.yml    # Conda environment definition
├── .gitignore         # Files & folders to exclude
└── LICENSE            # License (e.g., MIT)

🚀 Installation
git clone https://github.com/your-username/ECGConVT.git
cd ECGConVT

Create conda environment
conda env create -f environment.yml
conda activate ECGConVT

Install additional dependencies (if using pip)
pip install -r requirements.txt

🗂️ Data Preparation
Structure: Place your ECG images in data/raw/ with four subdirectories:
MI/
PMI/
AHB/
Normal/
Ignore data in git: Ensure data/raw/ and data/processed/ are listed in .gitignore.
Preprocessing: The loader (src/data_loader.py) will resize to 299×299, normalize, and split into train/val/test.

⚙️ Configuration
Edit config.yaml to adjust paths and hyperparameters:

data_dir: "data/raw"
img_size: [299, 299]
batch_size: 32
val_split: 0.2
test_split: 0.1
seed: 42
xcep_weights: "imagenet"
vit_url: "https://tfhub.dev/google/vit_base_patch16_224/feature_vector/1"
trainable_backbones: false
mlp_units: [512, 256]
dropout_rate: 0.5
optimizer: "Adam"
learning_rate: 1e-4
epochs: 200
checkpoint_dir: "checkpoints"
tensorboard_logdir: "logs/"
history_path: "reports/history.json"

🏃 Usage

Training
python src/train.py --config config.yaml

Evaluation
python src/evaluate.py --model_path checkpoints/best_model.h5 --test_dir data/raw


📓 Notebooks
Reuse the provided notebooks for data exploration, architecture visualization, and result analysis:
notebooks/01_data_exploration.ipynb
notebooks/02_model_architectures.ipynb
notebooks/03_results_analysis.ipynb

