# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains **MolBART (Chemformer)**, a BART transformer model pre-trained on molecular SMILES strings for chemical retrosynthesis tasks. It also includes **LLM4Chem**, which fine-tunes large language models on chemistry tasks.

The project has two main components:
1. **molbart**: BART-based transformer models for molecular tasks (reaction prediction, retrosynthesis, molecular optimization)
2. **LLM4Chem**: LoRA fine-tuning of LLMs (Mistral, Galactica, Llama2) for chemistry tasks

## Environment Setup

```bash
# Create conda environment
conda create --name molbart rdkit -c rdkit
conda activate molbart
conda install pytorch==1.8.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
conda install gcc_linux-64 gxx_linux-64 mpi4py
pip install -r requirements.txt

# Install pysmilesutils dependency
python -m pip install git+https://github.com/MolecularAI/pysmilesutils.git
```

## Running Scripts

### MolBART (Chemformer)

All molbart scripts are run as Python modules:

```bash
# Pre-training
python -m molbart.train <args>

# Fine-tuning on downstream tasks
python -m molbart.fine_tune \
  --dataset UsptoTXT_gamma \
  --data_path /path/to/data.txt \
  --model_path /path/to/pretrained.ckpt \
  --task backward_prediction \
  --epochs 10 \
  --lr 0.001 \
  --batch_size 128 \
  --acc_batches 4 \
  --gpus 1

# Evaluation
python -m molbart.evaluate \
  --model_path /path/to/model.ckpt \
  --dataset uspto_50 \
  --data_path /path/to/data.pickle \
  --task backward_prediction \
  --num_beams 10

# Generate predictions
python -m molbart.predict <args>

# Build tokeniser from dataset
python -m molbart.build_tokeniser <args>
```

### LLM4Chem

```bash
# Fine-tune LLM with LoRA
python LLM4Chem/finetune.py \
  --base_model osunlp/LlaSMol-Mistral-7B \
  --data_path /path/to/data \
  --output_dir checkpoint \
  --batch_size 512 \
  --micro_batch_size 4 \
  --num_epochs 3

# Generate predictions on dataset
python LLM4Chem/generate_on_dataset.py \
  --model_name osunlp/LlaSMol-Mistral-7B \
  --output_dir eval/output \
  --tasks "['forward_synthesis','retrosynthesis']"

# Extract predictions from output
python LLM4Chem/extract_prediction.py \
  --output_dir eval/output \
  --prediction_dir eval/prediction \
  --tasks "['forward_synthesis','retrosynthesis']"

# Compute metrics
python LLM4Chem/compute_metrics.py <args>
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest test/pre_train_model_test.py

# Tests require pytest-order for execution ordering
```

## Architecture

### MolBART Components

**Models** (`molbart/models/pre_train.py`):
- `_AbsTransformerModel`: Base PyTorch Lightning module for all models
- `BARTModel`: Standard BART encoder-decoder for seq2seq tasks
- `UnifiedModel`: Unified architecture variant
- Both inherit from `_AbsTransformerModel` and implement training/validation/test steps

**Data Pipeline** (`molbart/data/`):
- `_AbsDataset` classes: PyTorch Dataset subclasses for storing molecules/reactions (train/val/test splits)
- `_AbsDataModule` classes: PyTorch Lightning DataModules for augmentation, tokenization, and tensorization
- `TokenSampler`: Buckets sequences by length and samples different batch sizes per bucket to maintain consistent token counts across batches

**Tokenization** (`molbart/tokeniser.py`):
- `MolEncTokeniser`: BERT-style random masking, padding, uses `SMILESTokenizer` from pysmilesutils
- Tokenization regex: `\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]`

**Decoding** (`molbart/decoder.py`):
- Greedy and beam search implementations
- Batch decoding for speed (no caching, passes entire sequence through decoder each time)
- `DecodeSampler`: Main sampling class used by models

**LoRA Support**:
- `--lora`, `--encoder_lora`, `--decoder_lora` flags enable LoRA fine-tuning
- `--fix_encoder`, `--fix_decoder` flags freeze encoder/decoder
- `LinearWithLoRA` in `molbart/models/util.py` implements LoRA layers

### LLM4Chem Components

**Configuration** (`LLM4Chem/config.py`):
- `TASKS`: Supported chemistry tasks (forward_synthesis, retrosynthesis, molecule_captioning, property_prediction, etc.)
- `TASKS_GENERATION_SETTINGS`: Task-specific beam search and generation parameters
- `TASK_TAGS`: XML-style tags for output formatting (e.g., `<SMILES>`, `<NUMBER>`)
- `BASE_MODELS`: Mapping of LlaSMol models to their base models

**Training** (`LLM4Chem/finetune.py`):
- Uses PEFT library for LoRA fine-tuning
- `CustomTrainer` and `CustomDataCollator` handle training logic
- Supports distributed training with NCCL backend
- Uses 8-bit quantization with `adamw_bnb_8bit` optimizer

**Generation** (`LLM4Chem/generation.py`, `LLM4Chem/generate_on_dataset.py`):
- Task-specific beam search parameters from config
- Output extraction with XML tags via `CoreTagger`
- `GeneralPrompter` handles prompt formatting

**Metrics** (`LLM4Chem/utils/metrics.py`, `LLM4Chem/compute_metrics.py`):
- SMILES canonicalization via RDKit
- Task-specific evaluation metrics

## Key Design Patterns

**Multi-GPU Training**:
- Use `--gpus <num>` for training/fine-tuning
- Validation is disabled in DDP mode to avoid deadlocks
- `--train_tokens` must be None when using multiple GPUs

**Learning Rate Schedules**:
- `"cycle"`: OneCycleLR schedule
- `"transformer"`: Transformer-style schedule (requires `--warm_up_steps`)
- `"const"`: Constant learning rate

**Data Augmentation**:
- MolBART: `--augment` flag with options like "all", "None"
- Uses pysmilesutils `MolAugmenter` for SMILES augmentation
- `--aug_prob` controls augmentation probability

**Task Types**:
- `forward_prediction`: Reactants → Products
- `backward_prediction`: Products → Reactants (retrosynthesis)
- Dataset format depends on task (UsptoTXT, uspto_50, uspto_mixed, etc.)

**Vocabulary Files**:
- `config/vocabs/bart_vocab.txt`: Pre-training vocabulary
- `config/vocabs/bart_vocab_downstream.txt`: Fine-tuning vocabulary (default for downstream tasks)
- `config/vocabs/prop_bart_vocab.txt`: Regression modeling vocabulary with QSAR task tokens

## Configuration Management

**Vocabulary Configuration** (`config/vocabs/`):
All tokenizer vocabulary files are centralized in this directory. The code automatically references these paths via default constants in `molbart/util.py`, `molbart/fine_tune.py`, and `molbart/evaluate.py`.

**DeepSpeed Configuration** (`config/deepspeed/`):
`ds_config.json` configures DeepSpeed ZeRO Stage 2 optimization:
- FP16 training enabled
- CPU offload disabled
- Gradient bucketing for communication efficiency

## Dataset Conversion

`dataset_conversion/` contains scripts to convert various formats:
- `UsptoTXT/`: Convert JSONL/PBGZ to USPTO format
- `UsptoTXT_gamma/`: USPTO gamma variant conversion
- Output format: tab-separated reaction data

## Important Notes

- Models expect SMILES strings as input/output
- RDKit is required for molecule validation and canonicalization
- Pre-trained models and datasets available at: https://az.box.com/s/7eci3nd9vy0xplqniitpk02rbg9q2zcq
- Beam search typically uses 10 beams for evaluation (configurable via `--num_beams`)
- Sample datasets stored in `data/` directory (e.g., `data/retro_111.txt`)

## Scripts Organization

All executable scripts are organized in the `scripts/` directory:

**MolBART Scripts** (`scripts/molbart/`):

*Fine-tuning with USPTO_50 dataset*:
- `fine_tune_gamma1_baseline.sh`: Baseline with gamma=1
- `fine_tune_fix_decoder_with_layers.sh`: Fixed decoder + additional layers
- `fine_tune_fix_encoder_with_end_layer.sh`: Fixed encoder + end layer
- `fine_tune_fix_encoder.sh`: Fixed encoder only

*Fine-tuning with RetroLLM dataset* (data/retro_111.txt):
- `fine_tune_retrollm_gamma0.sh`: gamma=0
- `fine_tune_retrollm_gamma017.sh`: gamma=0.17
- `fine_tune_retrollm_gamma033.sh`: gamma=0.33

*Other MolBART scripts*:
- `pre_train.sh`: Pre-training script
- `eval_forward.sh`: Forward prediction evaluation
- `predict.sh`: Generate predictions
- `finetune_regression/`: Regression fine-tuning scripts and modules

**LLM4Chem Scripts** (`scripts/llm4chem/`):
- `run_fs.sh`: Run forward synthesis task
- `run_rs.sh`: Run retrosynthesis task

**Root-level Scripts**:
- `generate.sh`: Main generation script for LLM4Chem tasks
