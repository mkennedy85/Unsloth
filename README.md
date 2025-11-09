# Unsloth LLM Fine-Tuning Project

This project demonstrates various fine-tuning techniques using Unsloth.ai framework for modern LLM training tasks.

## Project Structure

```
.
├── notebooks/
│   ├── 01_full_finetuning_smollm2.ipynb
│   ├── 02_lora_finetuning_smollm2.ipynb
│   ├── 03_rl_preference_learning.ipynb
│   ├── 04_grpo_reasoning_model.ipynb
│   └── 05_continued_pretraining.ipynb
├── data/
│   └── README.md (dataset information)
├── outputs/
│   └── README.md (model outputs and checkpoints)
├── docs/
│   └── video_guidelines.md
└── README.md
```

## Overview

### Notebook 1: Full Fine-Tuning
- **Model**: SmolLM2-135M or Gemma-3-1B
- **Method**: Full parameter fine-tuning
- **Dataset**: Coding/Chat task dataset
- **Key Feature**: `load_in_4bit=False, full_finetuning=True`

### Notebook 2: LoRA Fine-Tuning
- **Model**: Same as Notebook 1 (SmolLM2-135M)
- **Method**: Parameter-efficient LoRA
- **Dataset**: Same dataset as Notebook 1
- **Key Feature**: LoRA adapters with rank configuration

### Notebook 3: Reinforcement Learning (DPO/ORPO)
- **Model**: Small instruction-tuned model
- **Method**: Direct Preference Optimization
- **Dataset**: Preference pairs (chosen/rejected)
- **Reference**: [Unsloth RL Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)

### Notebook 4: GRPO Reasoning Model
- **Model**: Base model for reasoning
- **Method**: Group Relative Policy Optimization
- **Dataset**: Problems with LLM-generated solutions
- **Reference**: [Unsloth R1 Reasoning](https://unsloth.ai/blog/r1-reasoning)

### Notebook 5: Continued Pretraining
- **Model**: Multilingual capable model
- **Method**: Continued pretraining
- **Task**: Teaching LLM a new language
- **Reference**: [Continued Pretraining Guide](https://docs.unsloth.ai/basics/continued-pretraining)

## Video Recording Guidelines

For each notebook, record a demonstration video covering:

1. **Introduction** (2-3 minutes)
   - Task overview
   - Model selection rationale
   - Dataset description

2. **Code Walkthrough** (10-15 minutes)
   - Installation and setup
   - Model loading with Unsloth
   - Dataset preparation and format
   - Training configuration
   - Fine-tuning process

3. **Results & Analysis** (5-7 minutes)
   - Training metrics
   - Sample outputs
   - Comparison with base model
   - Export process (Ollama/GGUF)

4. **Inference Demo** (3-5 minutes)
   - Loading fine-tuned model
   - Chat interface demonstration
   - Real-world examples

## Key Resources

- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Unsloth GitHub Notebooks](https://github.com/unslothai/notebooks/)
- [Kaggle Example](https://www.kaggle.com/code/kingabzpro/fine-tuning-llms-using-unsloth)
- [Medium Article - LoRA with Ollama](https://sarinsuriyakoon.medium.com/unsloth-lora-with-ollama-lightweight-solution-to-full-cycle-llm-development-edadb6d9e0f0)

## Supported Models

- Llama 3.1 (8B)
- Mistral NeMo (12B)
- Gemma 2 (9B)
- Phi-3.5 (mini)
- SmolLM2 (135M)
- Qwen2 (7B)
- TinyLlama

## Installation

```bash
# Install Unsloth
pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

# For Ollama export
pip install ollama
```

## Supported Platforms

This project works on any Jupyter-compatible platform:
- **Vertex AI Workbench** (GCP managed Jupyter) - Recommended
- **Kaggle Notebooks** (free GPU with P100/T4)
- **SageMaker Studio** (AWS managed Jupyter)
- **Local Jupyter** (with NVIDIA GPU)
- **JupyterHub** (multi-user environments)

## Chat Templates

Different models use different chat templates. Examples:

- **Llama 3**: `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>`
- **Mistral**: `<s>[INST] {prompt} [/INST]`
- **Gemma**: `<start_of_turn>user\n{prompt}<end_of_turn>`

## Export Options

1. **GGUF** for llama.cpp
2. **Ollama** for local deployment
3. **vLLM** for production serving
4. **HuggingFace** for sharing

## Next Steps

1. Set up your Vertex AI Workbench environment
2. Choose datasets for each task
3. Complete notebooks sequentially
4. Record demonstration videos
5. Host videos on GitHub or submit as required
