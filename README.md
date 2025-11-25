# LLM-based Chemical Retrosynthesis

## Overview

This repository is the final project for DS-GA 1011 Natural Language Processing with Representation Learning. 

It provides a unified framework for **chemical retrosynthesis prediction** combining two powerful approaches:

### MolBART (aka Chemformer)

A BART-based transformer model pre-trained on molecular SMILES strings, optimizing a denoising objective for improved generalization on downstream chemistry tasks including:
- Reaction prediction (forward synthesis)
- Retrosynthetic analysis (backward prediction)
- Molecular optimization
- Property prediction

### LLM4Chem

Large language model fine-tuning with LoRA (Low-Rank Adaptation) for chemistry tasks, supporting multiple foundation models:
- **LlaSMol-Mistral-7B**
- **LlaSMol-Galactica-6.7B**
- **LlaSMol-Llama2-7B**
- **LlaSMol-CodeLlama-7B**

---

## Features

- **Pre-trained Models**: Access to pre-trained BART models for molecular tasks
- **Efficient Fine-tuning**: LoRA-based parameter-efficient fine-tuning
- **Multiple Datasets**: Support for USPTO-50, USPTO-MIT, and custom datasets
- **Flexible Architecture**: Configurable encoder/decoder freezing and layer modifications
- **Multi-GPU Training**: Distributed training with PyTorch Lightning and DeepSpeed
- **Beam Search**: Advanced decoding strategies for improved prediction quality
- **Data Augmentation**: SMILES augmentation for robust training

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.1+ (for GPU support)
- Conda (recommended)

### Setup Environment

```bash
# Create conda environment with RDKit
conda create --name molbart python=3.8 rdkit -c rdkit
conda activate molbart

# Install PyTorch with CUDA support
conda install pytorch==1.8.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia

# Install additional dependencies
conda install gcc_linux-64 gxx_linux-64 mpi4py
pip install -r requirements.txt

# Install pysmilesutils
pip install git+https://github.com/MolecularAI/pysmilesutils.git

# Install the package
pip install -e .
```

### Verify Installation

```bash
python -c "import molbart; import torch; print('Installation successful!')"
pytest test/  # Run tests
```

---

## Quick Start

### 1. Pre-training a Model

```bash
python -m molbart.train \
  --dataset zinc \
  --data_path data/zinc_dataset.txt \
  --vocab_path config/vocabs/bart_vocab.txt \
  --d_model 512 \
  --num_layers 6 \
  --num_heads 8 \
  --d_feedforward 2048 \
  --batch_size 128 \
  --epochs 10 \
  --gpus 1
```

### 2. Fine-tuning for Retrosynthesis

```bash
# Using the provided script
bash scripts/molbart/fine_tune_retrollm_gamma0.sh

# Or run directly
python -m molbart.fine_tune \
  --dataset UsptoTXT_gamma \
  --data_path data/retro_111.txt \
  --model_path models/pretrained/combined.ckpt \
  --task backward_prediction \
  --epochs 100 \
  --lr 0.001 \
  --batch_size 128 \
  --gpus 1
```

### 3. Evaluation

```bash
python -m molbart.evaluate \
  --model_path models/finetuned/model.ckpt \
  --dataset uspto_50 \
  --data_path data/uspto_50.pickle \
  --task backward_prediction \
  --num_beams 10
```

### 4. Generate Predictions

```bash
python -m molbart.predict \
  --model_path models/finetuned/model.ckpt \
  --input_smiles "CCO" \
  --num_beams 10
```

---

## Project Structure

```
Chemical-Retrosynthesis/
├── config/                      # Configuration files
│   ├── vocabs/                 # Tokenizer vocabularies
│   └── deepspeed/              # DeepSpeed configurations
├── data/                        # Datasets (gitignored)
├── scripts/                     # Training and evaluation scripts
│   ├── molbart/               # MolBART experiment scripts
│   └── llm4chem/              # LLM4Chem scripts
├── molbart/                     # Core MolBART package
│   ├── models/                # Model implementations
│   ├── data/                  # Data loading and processing
│   ├── tokeniser.py           # SMILES tokenization
│   ├── decoder.py             # Beam search and decoding
│   ├── train.py               # Pre-training script
│   ├── fine_tune.py           # Fine-tuning script
│   └── evaluate.py            # Evaluation script
├── LLM4Chem/                    # LLM4Chem submodule
├── dataset_conversion/          # Dataset format converters
└── test/                        # Unit tests
```

---

## Configuration

### Vocabulary Files

Located in `config/vocabs/`:
- `bart_vocab.txt`: Pre-training vocabulary
- `bart_vocab_downstream.txt`: Fine-tuning vocabulary (default)
- `prop_bart_vocab.txt`: Property prediction vocabulary with QSAR tokens

### Training Configurations

**Learning Rate Schedules**:
- `cycle`: OneCycleLR schedule (recommended for fine-tuning)
- `transformer`: Transformer-style schedule with warmup
- `const`: Constant learning rate

**Model Architectures**:
- `bart`: Standard BART encoder-decoder
- `unified`: Unified architecture variant

**Advanced Options**:
- `--lora`: Enable LoRA fine-tuning
- `--fix_encoder`: Freeze encoder weights
- `--fix_decoder`: Freeze decoder weights
- `--gamma`: Loss weighting parameter for RetroLLM

---

## Experiment Scripts

### MolBART Experiments

**USPTO-50 Dataset**:
- `fine_tune_gamma1_baseline.sh`: Baseline configuration
- `fine_tune_fix_decoder_with_layers.sh`: Frozen decoder + additional layers
- `fine_tune_fix_encoder_with_end_layer.sh`: Frozen encoder + end layer
- `fine_tune_fix_encoder.sh`: Frozen encoder only

**RetroLLM Dataset** (testing gamma values):
- `fine_tune_retrollm_gamma0.sh`: γ = 0.0
- `fine_tune_retrollm_gamma017.sh`: γ = 0.17
- `fine_tune_retrollm_gamma033.sh`: γ = 0.33

### LLM4Chem

```bash
# Fine-tune with LoRA
python LLM4Chem/finetune.py \
  --base_model osunlp/LlaSMol-Mistral-7B \
  --data_path data/chemistry_dataset.json \
  --output_dir checkpoints/llm4chem \
  --batch_size 512 \
  --micro_batch_size 4 \
  --num_epochs 3

# Generate predictions
python LLM4Chem/generate_on_dataset.py \
  --model_name osunlp/LlaSMol-Mistral-7B \
  --output_dir eval/output \
  --tasks "['forward_synthesis','retrosynthesis']"
```

---

## Multi-GPU Training

### Using PyTorch Lightning DDP

```bash
python -m molbart.fine_tune \
  --gpus 4 \
  --batch_size 32 \
  --acc_batches 4 \
  # ... other arguments
```

### Using DeepSpeed

```bash
python -m molbart.train \
  --deepspeed config/deepspeed/ds_config.json \
  --gpus 8 \
  # ... other arguments
```

**Note**: Validation is disabled in DDP mode to prevent deadlocks. The `--train_tokens` parameter must be `None` when using multiple GPUs.

---

## Pre-trained Models & Datasets

Download pre-trained models and datasets from:
- 📦 [Box Repository](https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq)

**Available Models**:
- Combined pre-trained BART (1M steps)
- Fine-tuned models for USPTO-50
- Task-specific fine-tuned models

---

## Advanced Usage

### Custom Dataset

```python
from molbart.data.datasets import ReactionDataset
from molbart.data.datamodules import FineTuneReactionDataModule

# Load your dataset
dataset = ReactionDataset("path/to/data.txt")

# Create data module
dm = FineTuneReactionDataModule(
    dataset=dataset,
    tokeniser=tokeniser,
    batch_size=64,
    max_seq_len=512
)
```

### Custom Training Loop

```python
import pytorch_lightning as pl
from molbart.models.pre_train import BARTModel

# Initialize model
model = BARTModel(...)

# Setup trainer
trainer = pl.Trainer(
    gpus=1,
    max_epochs=50,
    precision=16
)

# Train
trainer.fit(model, datamodule=dm)
```

---

## Performance

### USPTO-50 Benchmark (Top-1 Accuracy)

| Model | Accuracy |
|-------|----------|
| MolBART (baseline) | 58.3% |
| MolBART + LoRA | 59.7% |
| LlaSMol-Mistral-7B | 62.1% |

*Results may vary based on hyperparameters and training configuration*

---

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test/pre_train_model_test.py

# Run with coverage
pytest --cov=molbart test/
```

</div>
