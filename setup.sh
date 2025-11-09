#!/bin/bash

# Setup script for Jupyter environments with per-notebook kernels
# Compatible with: Vertex AI Workbench, SageMaker, Kaggle, Local Jupyter
# Run this after cloning the repository

echo "=========================================="
echo "Unsloth Fine-Tuning Project Setup"
echo "=========================================="
echo ""

# Check if running on GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo ""
else
    echo "⚠️  Warning: No GPU detected. Training will be very slow."
    echo "   Make sure you're using a GPU-enabled instance."
    echo ""
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p outputs
mkdir -p data
mkdir -p models
mkdir -p kernels
echo "✓ Directories created"
echo ""

# Function to create a kernel for a notebook
create_kernel() {
    local kernel_name=$1
    local display_name=$2
    local notebook_file=$3

    echo "Setting up kernel: $display_name"

    # Create virtual environment for this notebook
    python3 -m venv "kernels/$kernel_name"
    source "kernels/$kernel_name/bin/activate"

    # Install ipykernel
    pip install --quiet ipykernel

    # Install Unsloth and dependencies
    pip install --quiet "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
    pip install --quiet --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

    # Install additional dependencies from requirements.txt
    if [ -f requirements.txt ]; then
        pip install --quiet -r requirements.txt
    fi

    # Register the kernel with Jupyter
    python3 -m ipykernel install --user --name="$kernel_name" --display-name="$display_name"

    echo "✓ Kernel '$display_name' created"

    deactivate
}

# Prompt user for setup type
echo "Choose setup type:"
echo "1. Quick setup (single shared kernel for all notebooks)"
echo "2. Full setup (separate kernel per notebook - recommended)"
echo ""
read -p "Enter choice (1 or 2): " setup_choice
echo ""

if [ "$setup_choice" == "2" ]; then
    echo "Creating separate kernels for each notebook..."
    echo "This will take 10-15 minutes but ensures isolated environments."
    echo ""

    # Create kernel for each notebook
    create_kernel "unsloth-01-full-ft" "Unsloth 01: Full Fine-Tuning" "notebooks/01_full_finetuning_smollm2.ipynb"
    echo ""

    create_kernel "unsloth-02-lora" "Unsloth 02: LoRA Fine-Tuning" "notebooks/02_lora_finetuning_smollm2.ipynb"
    echo ""

    create_kernel "unsloth-03-dpo-orpo" "Unsloth 03: DPO/ORPO" "notebooks/03_rl_preference_learning.ipynb"
    echo ""

    echo "✓ All kernels created!"
    echo ""
    echo "Kernel names:"
    echo "  - Unsloth 01: Full Fine-Tuning"
    echo "  - Unsloth 02: LoRA Fine-Tuning"
    echo "  - Unsloth 03: DPO/ORPO"
    echo ""

else
    # Quick setup - single kernel
    echo "Creating shared kernel for all notebooks..."
    echo ""

    # Install Unsloth
    echo "Installing Unsloth and dependencies..."
    pip install --quiet "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
    pip install --quiet --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

    # Install additional dependencies
    if [ -f requirements.txt ]; then
        pip install --quiet -r requirements.txt
    fi

    echo "✓ Packages installed in current environment"
    echo ""
fi

# Test installation
echo "Testing installation..."
python3 << EOF
try:
    from unsloth import FastLanguageModel
    import torch
    print("✓ Unsloth imported successfully")
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Setup complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    if [ "$setup_choice" == "2" ]; then
        echo "1. Open notebooks/01_full_finetuning_smollm2.ipynb"
        echo "2. Select kernel: 'Unsloth 01: Full Fine-Tuning'"
        echo "3. Run cells sequentially"
        echo ""
        echo "Note: Each notebook has its own isolated kernel."
        echo "Select the matching kernel when opening each notebook."
    else
        echo "1. Open notebooks/01_full_finetuning_smollm2.ipynb"
        echo "2. Make sure Jupyter kernel is set to Python 3"
        echo "3. Run cells sequentially"
    fi
    echo ""
    echo "Available notebooks:"
    echo "- 01_full_finetuning_smollm2.ipynb (✓ Ready)"
    echo "- 02_lora_finetuning_smollm2.ipynb (✓ Ready)"
    echo "- 03_rl_preference_learning.ipynb (✓ Ready)"
    echo "- 04_grpo_reasoning_model.ipynb (Coming soon)"
    echo "- 05_continued_pretraining.ipynb (Coming soon)"
    echo ""

    if [ "$setup_choice" == "2" ]; then
        echo "To remove all kernels later, run:"
        echo "  jupyter kernelspec remove unsloth-01-full-ft"
        echo "  jupyter kernelspec remove unsloth-02-lora"
        echo "  jupyter kernelspec remove unsloth-03-dpo-orpo"
        echo ""
    fi
else
    echo ""
    echo "=========================================="
    echo "✗ Setup failed"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    echo "Common issues:"
    echo "- No GPU available"
    echo "- Python version < 3.8"
    echo "- CUDA driver mismatch"
    echo ""
    exit 1
fi
