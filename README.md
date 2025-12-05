# IDOL API Setup

Instant Photorealistic 3D Human Creation from a Single Image using IDOL.

## Requirements

- NVIDIA GPU with 24GB+ VRAM (recommended)
- CUDA 11.8 compatible drivers
- Python 3.10+
- SMPL-X registration (for full functionality)

## Quick Start

### One-Click Setup

```bash
chmod +x setup.sh
./setup.sh
```

### SMPL-X Setup (Required)

Register at: https://smpl-x.is.tue.mpg.de/

Then run:
```bash
bash scripts/fetch_template.sh
```

### Usage

```bash
# Reconstruction mode
python run_demo.py --render_mode reconstruct

# Animation mode
python run_demo.py --render_mode novel_pose

# 360-degree view
python run_demo.py --render_mode novel_pose_A

# Start API server
python api.py
```

## API Endpoints

- `GET /health` - Health check
- `POST /generate/upload` - Upload image for processing
- `GET /download/{filename}` - Download generated files

## Render Modes

- `reconstruct` - Reconstruct the input image
- `novel_pose` - Generate with novel poses (animation)
- `novel_pose_A` - Generate 360-degree view with A-pose

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

MIT License
