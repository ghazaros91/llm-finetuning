# LLM Fine-Tuning Pipeline

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-supported-yellow)](https://huggingface.co/docs/transformers)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20support-red)](https://github.com/huggingface/peft)

A comprehensive pipeline for fine-tuning Large Language Models with instruction datasets, featuring:
- LoRA (Low-Rank Adaptation) training
- Dataset preprocessing and caching
- Instruction following fine-tuning

## Features
- **Efficient Fine-Tuning**:
  - LoRA with configurable rank and alpha parameters
  - Gradient accumulation for memory efficiency
- **Dataset Processing**:
  - Automatic tokenization and padding
  - Caching of processed datasets
  - Support for instruction-response formats
- **Flexible Configuration**:
  - YAML-based configuration for models and datasets
  - Adjustable sequence lengths
  - Multiple training parameters

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (recommended)
- PyTorch with CUDA support

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ghazaros91/llm-fine-tuning.git
   cd llm-fine-tuning
