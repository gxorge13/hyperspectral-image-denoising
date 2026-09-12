# Hyperspectral Image Denoising

An experimental **PyTorch denoising pipeline** for hyperspectral image cubes. A RIDNet-style network uses residual connections and enhancement attention modules to reconstruct clean patches from inputs with synthetic Gaussian noise.

**Stack:** Python · PyTorch · NumPy · TorchMetrics · scikit-image · Matplotlib

## How it works

1. Load a hyperspectral cube from a NumPy file and normalize each spectral band to `[0, 1]`.
2. Sample spatial patches and add Gaussian noise during data loading.
3. Train a residual network containing four enhancement attention modules, with dilated convolutions and channel attention.
4. Evaluate reconstruction using **PSNR** and per-band **SSIM**, then plot the training loss and a noisy / reconstructed / clean comparison.

## Implementation

| File | Purpose |
| --- | --- |
| [models/ModelEAM.py](models/ModelEAM.py) | `EAM` attention module and `RIDNet` network |
| [trainnp.py](trainnp.py) | Dataset preparation, training, evaluation, and visualization |
| [yer.py](yer.py) | Small utility to inspect the input array's size |
| [checkpoints/](checkpoints/) | Saved model weights from earlier experiments |

The training script selects Apple MPS, CUDA, or CPU in that order. Its current defaults are 64 × 64 patches, batch size 4, 200 epochs, Gaussian noise standard deviation 0.05, Adam optimization, L1 loss, and exponential learning-rate decay.

## Setup

```bash
git clone https://github.com/gxorge13/hyperspectral-image-denoising.git
cd hyperspectral-image-denoising
python -m venv .venv
source .venv/bin/activate
python -m pip install torch numpy matplotlib torchmetrics scikit-image
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. Dependency versions are not pinned; the original environment has not been reconstructed.

### Prepare the data

Place the input cube at `data/indian_pine_array.npy`, relative to the repository root. This file is **not included**.

- Expected array layout: `(height, width, spectral_bands)`.
- Height and width must each be at least 64 for the current patch size.
- `trainnp.py` constructs the model with **200 input and output bands**. Match the data to that setting or update both channel counts. The model class's standalone defaults differ from the training script.

Once compatible data and dependencies are available, run from the repository root:

```bash
python trainnp.py
```

This starts the full training loop and displays plots; it is not a lightweight inference demo. Checkpoint saving is currently commented out in the training script.

## Evaluation limits

This is a research prototype using **synthetic Gaussian noise**. It does not establish performance on measured sensor dark noise.

The current train/validation split divides dataset indices, but patch sampling ignores those indices and draws from the same source cube. The two subsets can therefore overlap spatially. Treat these metrics as within-cube exploratory results, not an independent held-out benchmark.

Saved checkpoint filenames contain historical PSNR values, but no independently reproduced benchmark is claimed here. End-to-end training has not been rerun because the required input cube is absent.

## Attribution

The model follows a RIDNet-style residual denoising design. This repository contains an experimental implementation and adaptation, rather than a claim to have originated that architecture.
