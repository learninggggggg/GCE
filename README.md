# GEC: A Global and Emotion-Consistent Context Modeling Framework for Suicide Risk Detection

[//]: # (<a href='#'><img src='https://img.shields.io/badge/Paper-PDF-blue'></a>)

[//]: # (<a href='#'><img src='https://img.shields.io/badge/Conference-EMNLP-orange'></a>)
<a href='#'><img src='https://img.shields.io/badge/Language-Python-yellow'></a>
<a href='#'><img src='https://img.shields.io/badge/Framework-PyTorch-red'></a>

[Zhuping Ding](), [Yongpan Sheng](),[Lirong He](), [Yiran Wang](), and [Ming Liu]()

## Release
- [11/25] Initial release of GEC codebase for suicide risk detection.

## Contents
- [Release](#release)
- [Contents](#contents)
- [Overview](#overview)
- [Installation](#insta)

Dataset

Quick Start

Training

Inference

Citation

License

## Overview
GEC is a novel framework for suicide risk detection that incorporates both global context modeling and emotion-consistent representation learning. This approach addresses the critical need for accurate and sensitive detection of suicide risk in online text content.

Key Features:

Global Context Modeling: Captures long-range dependencies and contextual information

Emotion-Consistent Learning: Ensures emotional coherence across the text

Multi-scale Analysis: Integrates both local and global semantic information

Robust Performance: State-of-the-art results on suicide risk detection benchmarks

## Installation
Clone this repository:

```
git clone https://github.com/your-username/GEC-Suicide-Risk-Detection.git
cd GEC-Suicide-Risk-Detection
```

Create and activate a conda environment:
```
conda create -n gec python=3.8 -y
conda activate gec
```
Install required dependencies:

```
pip install -r requirements.txt
```

Install PyTorch (choose the appropriate version for your CUDA setup):
```
# For CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
## Dataset
Data Preparation
Download the suicide risk detection dataset

Preprocess the data using our provided scripts:

bash
python scripts/preprocess_data.py --input_path /path/to/raw_data --output_path data/processed/
Data Format
The processed data should be in the following format:

json
{
  "text": "sample text content",
  "label": 0,
  "emotion": "sadness",
  "metadata": {...}
}
Quick Start
Basic Usage
Run the demo with pre-trained weights:

bash
python demo.py --model_path checkpoints/pretrained_gec --text "Your input text here"
Training from Scratch
bash
python train.py --config configs/train_config.yaml --data_path data/processed/ --output_dir outputs/
Evaluation
bash
python evaluate.py --model_path outputs/best_model/ --test_data data/processed/test.json
Training
Single GPU Training
bash
python train.py \
    --model_name bert-base-uncased \
    --train_file data/train.json \
    --val_file data/val.json \
    --output_dir models/gec_model \
    --num_epochs 10 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --max_length 256
Multi-GPU Training
bash
torchrun --nproc_per_node=4 train.py \
    --model_name bert-base-uncased \
    --train_file data/train.json \
    --val_file data/val.json \
    --output_dir models/gec_model \
    --num_epochs 10 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --max_length 256
Hyperparameter Tuning
We provide a configuration file for easy hyperparameter tuning:

bash
python train.py --config configs/hyperparameter_tuning.yaml
Inference
Batch Inference
bash
python inference.py \
    --model_path models/gec_model \
    --input_file data/test.json \
    --output_file results/predictions.json
Real-time Inference
python
from gec_model import GECModel

model = GECModel.from_pretrained('models/gec_model')
result = model.predict("Sample text for suicide risk detection")
print(f"Risk level: {result['risk_level']}, Confidence: {result['confidence']}")
Web Demo
Launch an interactive web interface:

bash
python app.py --model_path models/gec_model --port 7860
Configuration
Model Configuration
Key parameters in configs/model_config.yaml:

yaml
model:
  name: "bert-base-uncased"
  hidden_size: 768
  num_labels: 3
  dropout: 0.1
  
training:
  batch_size: 32
  learning_rate: 2e-5
  warmup_steps: 1000
  max_epochs: 10
  
data:
  max_length: 256
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
Citation
If you use GEC in your research, please cite our paper:

bibtex
@article{ding2024gec,
  title={GEC: A Global and Emotion-Consistent Context Modeling Framework for Suicide Risk Detection},
  author={Ding, Zhuping and Sheng, Yongpan and He, Lirong and Wang, Yiran and Liu, Ming},
  journal={Proceedings of the Conference on Empirical Methods in Natural Language Processing},
  year={2024}
}
License
https://img.shields.io/badge/License-MIT-yellow.svg

This project is licensed under the MIT License - see the LICENSE file for details.

Important Note: This software is intended for research purposes only. It should not be used as a substitute for professional mental health advice, diagnosis, or treatment. If you or someone you know is experiencing suicidal thoughts, please contact a mental health professional or emergency services immediately.

Contact
For questions about this codebase, please open an issue on GitHub or contact [Your Name] at [your.email@institution.edu].