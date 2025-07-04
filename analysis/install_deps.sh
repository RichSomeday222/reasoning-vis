echo "Installing dependencies with persistence..."

if python -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed"
else
    echo "Installing PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

if python -c "import transformers" 2>/dev/null; then
    echo "Transformers already installed"
else
    echo "Installing Transformers..."
    pip install transformers accelerate bitsandbytes huggingface_hub psutil
fi

export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface

echo "Dependencies ready!"
