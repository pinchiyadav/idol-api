# IDOL API Setup

Instant Photorealistic 3D Human Creation from a Single Image using IDOL.

## Requirements

- NVIDIA GPU with 24GB+ VRAM (recommended)
- CUDA 12.x compatible drivers
- Python 3.10+
- **SMPL-X models** (requires registration - see below)

## Quick Start

### One-Click Setup

```bash
chmod +x setup.sh
./setup.sh
```

### ⚠️ SMPL-X Setup (REQUIRED)

IDOL requires SMPL-X body models which must be downloaded separately due to licensing:

1. **Register** at: https://smpl-x.is.tue.mpg.de/
2. **Download**: `models_smplx_v1_1.zip`
3. **Extract** to: `IDOL/lib/models/deformers/smplx/SMPLX/`

Required files:
```
IDOL/lib/models/deformers/smplx/SMPLX/
├── SMPLX_NEUTRAL.pkl
├── SMPLX_MALE.pkl
└── SMPLX_FEMALE.pkl
```

Or run the interactive script:
```bash
cd IDOL
bash scripts/fetch_template.sh
```

### Usage

```bash
cd IDOL

# Reconstruction mode (input → 3D reconstruction)
python run_demo.py --input_path image.png --render_mode reconstruct

# Animation mode (input → animated 3D)
python run_demo.py --input_path image.png --render_mode novel_pose

# 360-degree A-pose view
python run_demo.py --input_path image.png --render_mode novel_pose_A

# Start API server
python api.py
```

## API Endpoints

- `GET /health` - Health check
- `POST /generate/upload` - Upload image for processing
- `GET /download/{filename}` - Download generated files

## Render Modes

| Mode | Description |
|------|-------------|
| `reconstruct` | Reconstruct the input image as 3D |
| `novel_pose` | Generate with novel poses (animation) |
| `novel_pose_A` | Generate 360-degree view with A-pose |

## Model Downloads

The setup script downloads:
- IDOL checkpoint: ~1.4GB
- Sapiens model: ~3.8GB

## Citation

```bibtex
@article{zhuang2024idolinstant,                
  title={IDOL: Instant Photorealistic 3D Human Creation from a Single Image}, 
  author={Yiyu Zhuang and others},
  journal={arXiv preprint arXiv:2412.14963},
  year={2024}
}
```

## License

Please refer to the original IDOL repository for licensing terms.
SMPL-X models are subject to their own license agreement.
