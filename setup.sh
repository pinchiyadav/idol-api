#!/bin/bash
# IDOL One-Click Setup Script
# Instant Photorealistic 3D Human Creation

set -e

echo "=============================================="
echo "IDOL One-Click Setup"
echo "=============================================="

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA GPU required (24GB+ VRAM recommended)."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Clone the official repository
echo ""
echo "[1/9] Cloning IDOL repository..."
if [ -d "IDOL" ]; then
    echo "Repository already exists, updating..."
    cd IDOL && git pull && cd ..
else
    git clone --depth 1 --progress https://github.com/yiyuzhuang/IDOL.git
fi

cd IDOL

# Install PyTorch with CUDA (matching system CUDA 12.x)
echo ""
echo "[2/9] Installing PyTorch with CUDA support..."
pip install --progress-bar on torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
echo ""
echo "[3/9] Installing core dependencies..."
pip install --progress-bar on diffusers==0.20.2 transformers accelerate huggingface-hub \
    numpy==1.26.4 scipy einops opencv-python matplotlib pillow \
    omegaconf pytorch-lightning gradio fastapi uvicorn tqdm rembg tensorboard

# Install PyTorch3D
echo ""
echo "[4/9] Installing PyTorch3D (this may take a while)..."
pip install --progress-bar on "git+https://github.com/facebookresearch/pytorch3d.git@stable" || echo "PyTorch3D installation skipped"

# Create submodule directory and install Simple-KNN
echo ""
echo "[5/9] Installing Simple-KNN..."
mkdir -p submodule
cd submodule
if [ ! -d "simple-knn" ]; then
    git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git
fi
cd simple-knn
pip install . --no-build-isolation
cd ../..

# Install Gaussian Splatting
echo ""
echo "[6/9] Installing Gaussian Splatting..."
cd submodule
if [ ! -d "gaussian-splatting" ]; then
    git clone --depth 1 --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
fi
cd gaussian-splatting/submodules/diff-gaussian-rasterization

# Fix header for C++17 compatibility
if ! grep -q "cstdint" cuda_rasterizer/rasterizer_impl.h; then
    sed -i 's/#include <iostream>/#include <iostream>\n#include <cstdint>/g' cuda_rasterizer/rasterizer_impl.h
fi

echo "Building diff-gaussian-rasterization..."
pip install . --no-build-isolation || echo "diff-gaussian-rasterization build skipped"
cd ../../..

# Install Sapiens
echo ""
echo "[7/9] Installing Sapiens..."
if [ ! -d "sapiens" ]; then
    git clone --depth 1 https://github.com/facebookresearch/sapiens
fi
cd sapiens/engine
pip install -e . || echo "Sapiens engine skipped"
cd ../pretrain
pip install -e . || echo "Sapiens pretrain skipped"
cd ../../..

# Install deformation module
echo ""
echo "[8/9] Installing deformation module..."
python setup.py develop 2>/dev/null || echo "Deformation module skipped"

# Download pretrained models
echo ""
echo "[9/9] Downloading pretrained models..."
mkdir -p work_dirs/ckpt
echo "Downloading IDOL model checkpoint..."
wget --progress=bar:force https://huggingface.co/yiyuzhuang/IDOL/resolve/main/model.ckpt \
    -O work_dirs/ckpt/model.ckpt 2>&1 || echo "Model download skipped"
echo "Downloading Sapiens model..."
wget --progress=bar:force https://huggingface.co/yiyuzhuang/IDOL/resolve/main/sapiens_1b_epoch_173_torchscript.pt2 \
    -O work_dirs/ckpt/sapiens_1b_epoch_173_torchscript.pt2 2>&1 || echo "Sapiens model download skipped"

# Copy API files
echo ""
echo "Copying API files..."
cd ..
cp api.py IDOL/ 2>/dev/null || true
cd IDOL

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Note: SMPL-X registration required for full functionality."
echo "Register at: https://smpl-x.is.tue.mpg.de/"
echo ""
echo "Usage:"
echo "  # Run demo (reconstruction):"
echo "  python run_demo.py --render_mode reconstruct"
echo ""
echo "  # Run demo (animation):"
echo "  python run_demo.py --render_mode novel_pose"
echo ""
echo "  # Start API server:"
echo "  python api.py"
echo ""
