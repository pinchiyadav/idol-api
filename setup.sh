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
    git clone --depth 1 https://github.com/yiyuzhuang/IDOL.git
fi

cd IDOL

# Install PyTorch with CUDA
echo ""
echo "[2/9] Installing PyTorch with CUDA support..."
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118 -q

# Install core dependencies
echo ""
echo "[3/9] Installing core dependencies..."
pip install diffusers==0.20.2 transformers accelerate huggingface-hub \
    numpy==1.26.4 scipy einops opencv-python matplotlib pillow \
    omegaconf pytorch-lightning gradio fastapi uvicorn tqdm rembg tensorboard -q

# Install PyTorch3D
echo ""
echo "[4/9] Installing PyTorch3D..."
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7" -q || true

# Create submodule directory
echo ""
echo "[5/9] Installing Simple-KNN..."
mkdir -p submodule
cd submodule
git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git 2>/dev/null || true
cd simple-knn
pip install . --no-build-isolation -q
cd ../..

# Install Gaussian Splatting
echo ""
echo "[6/9] Installing Gaussian Splatting..."
cd submodule
git clone --depth 1 --recursive https://github.com/graphdeco-inria/gaussian-splatting.git 2>/dev/null || true
cd gaussian-splatting/submodules/diff-gaussian-rasterization

# Fix header for C++17 compatibility
sed -i 's/#include <iostream>/#include <iostream>\n#include <cstdint>/g' cuda_rasterizer/rasterizer_impl.h 2>/dev/null || true

TORCH_CUDA_ARCH_LIST="8.6" pip install . --no-build-isolation -q || true
cd ../../..

# Install Sapiens
echo ""
echo "[7/9] Installing Sapiens..."
git clone --depth 1 https://github.com/facebookresearch/sapiens 2>/dev/null || true
cd sapiens/engine
pip install -e . -q || true
cd ../pretrain
pip install -e . -q || true
cd ../../..

# Install deformation module
echo ""
echo "[8/9] Installing deformation module..."
python setup.py develop 2>/dev/null || true

# Download pretrained models
echo ""
echo "[9/9] Downloading pretrained models..."
mkdir -p work_dirs/ckpt
wget -q https://huggingface.co/yiyuzhuang/IDOL/resolve/main/model.ckpt \
    -O work_dirs/ckpt/model.ckpt 2>/dev/null || echo "Model download skipped"
wget -q https://huggingface.co/yiyuzhuang/IDOL/resolve/main/sapiens_1b_epoch_173_torchscript.pt2 \
    -O work_dirs/ckpt/sapiens_1b_epoch_173_torchscript.pt2 2>/dev/null || echo "Sapiens model download skipped"

# Copy API files
cp ../api.py . 2>/dev/null || true

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
