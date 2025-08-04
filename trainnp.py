import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
import time
import random
import os

# SSIM on torch metrics wont work for some reason, so we use scikit-image
from skimage.metrics import structural_similarity as compare_ssim

from torchmetrics.image import PeakSignalNoiseRatio

# Import your RIDNet model
from models.ModelEAM import RIDNet

# For reproducibility
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

##############################
# 1. PATCH-BASED DATASET DEFINITION
##############################
class IndianPinePatchDataset(Dataset):
    def __init__(self, npy_file, patch_size=64, noise_std=0.05, num_patches=10000):
        """
        Loads the entire hyperspectral cube from the .npy file, performs per-band normalization,
        and stores it. Gaussian noise will be added on the fly in __getitem__.

        Args:
            npy_file (str): Path to the .npy file.
            patch_size (int): Height and width of each square patch.
            noise_std (float): Standard deviation of the Gaussian noise.
            num_patches (int): Number of patches to sample per epoch.
        """
        # Load the .npy file; expected shape: (height, width, bands)
        raw_image = np.load(npy_file).astype(np.float32)  # (H, W, B)

        # Transpose to (B, H, W)
        image = raw_image.transpose(2, 0, 1)  # (B, H, W)

        # Per-band normalization to [0, 1]
        band_mins = image.min(axis=(1, 2), keepdims=True)
        band_maxs = image.max(axis=(1, 2), keepdims=True)
        image = (image - band_mins) / (band_maxs - band_mins + 1e-8)

        self.image = image  # (B, H, W)
        self.bands, self.H, self.W = image.shape
        self.patch_size = patch_size
        self.noise_std = noise_std
        self.num_patches = num_patches

    def __len__(self):
        # We simulate having 'num_patches' distinct samples per epoch
        return self.num_patches

    def __getitem__(self, idx):
        """
        Returns a tuple (noisy_patch, clean_patch) where:
          - clean_patch is a randomly sampled square patch from the normalized image.
          - noisy_patch is computed on the fly by adding Gaussian noise to that patch.
        """
        # Randomly select top-left corner for the patch
        y = random.randint(0, self.H - self.patch_size)
        x = random.randint(0, self.W - self.patch_size)

        clean_patch = self.image[:, y : y + self.patch_size, x : x + self.patch_size]  # (B, patch, patch)

        # Add Gaussian noise to each band
        noisy_patch = clean_patch + np.random.normal(0, self.noise_std, clean_patch.shape)
        noisy_patch = np.clip(noisy_patch, 0.0, 1.0)

        # Convert to PyTorch tensors
        clean_patch = torch.from_numpy(clean_patch).float()
        noisy_patch = torch.from_numpy(noisy_patch).float()
        return noisy_patch, clean_patch


##############################
# 2. EVALUATION FUNCTION (with PSNR + manual SSIM)
##############################
def evaluate(model, dataloader, device, psnr_metric):
    model.eval()
    psnr_metric.reset()

    # We'll accumulate SSIM and count how many (patch, channel) pairs we processed
    total_ssim = 0.0
    ssim_count = 0

    with torch.no_grad():
        for noisy, clean in dataloader:
            # Move to device
            noisy = noisy.to(device)
            clean = clean.to(device)

            # Forward pass
            outputs = model(noisy)

            # Update PSNR from TorchMetrics (it expects full tensors in [0,1])
            psnr_metric.update(outputs, clean)

            # Convert outputs & clean to NumPy (CPU) for SSIM computation
            # shapes: (batch_size, bands, H, W)
            out_np = outputs.cpu().numpy()
            clean_np = clean.cpu().numpy()

            batch_size, B, H, W = out_np.shape

            # Compute SSIM for each patch in batch, each channel:
            for i in range(batch_size):
                for c in range(B):
                    # out_np[i, c] is a (H, W) array, same for clean_np[i, c]
                    ssim_val = compare_ssim(
                        clean_np[i, c],     # ground-truth band as 2D
                        out_np[i, c],       # reconstructed band as 2D
                        data_range=1.0
                    )
                    total_ssim += ssim_val
                    ssim_count += 1

    # Final metrics
    psnr_val = psnr_metric.compute().item()
    if ssim_count > 0:
        ssim_val = total_ssim / ssim_count
    else:
        ssim_val = 0.0

    return psnr_val, ssim_val


##############################
# 3. MAIN TRAINING LOOP
##############################
def main():
    # 1) Device Selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 2) Hyperparameters
    lr = 1e-3
    epochs = 200
    patch_size = 64
    batch_size = 4
    noise_std = 0.05

    # 3) Create dataset & DataLoaders
    dataset = IndianPinePatchDataset(
        npy_file="data/indian_pine_array.npy",
        patch_size=patch_size,
        noise_std=noise_std,
        num_patches=10000,
    )

    # 80/20 train/validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # 4) Initialize model, optimizer, loss, and PSNR metric
    model = RIDNet(in_channels=200, out_channels=200).to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    # Note: We no longer use TorchMetrics’ SSIM directly, because
    # the installed version doesn’t accept `channel_axis` or `multichannel`.
    # We’ll compute SSIM manually via scikit-image.

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()  # L1 (MAE) for denoising

    # Gentle LR decay so it doesn’t drop too fast over 200 epochs
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    # For tracking
    train_losses = []
    # best_val_psnr = 0.0

    # Create a folder for checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    # 5) Training Loop
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for noisy_inputs, clean_targets in train_loader:
            noisy_inputs = noisy_inputs.to(device)
            clean_targets = clean_targets.to(device)

            optimizer.zero_grad()
            outputs = model(noisy_inputs)
            loss = criterion(outputs, clean_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 6) Validation: compute PSNR + manual SSIM
        val_psnr, val_ssim = evaluate(model, val_loader, device, psnr_metric)

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val PSNR: {val_psnr:.4f} | Val SSIM: {val_ssim:.4f}"
        )
        print("Time elapsed: {:.2f} seconds".format(time.time() - start_time))

        # 7) Checkpoint if PSNR improved
        # if val_psnr > best_val_psnr:
        #     best_val_psnr = val_psnr
        #     ckpt_path = f"checkpoints/ridnet_epoch{epoch}_psnr{val_psnr:.2f}.pth"
        #     # torch.save(model.state_dict(), ckpt_path)
        #     print(f"  → Saved new best model to {ckpt_path}")

        scheduler.step()  # Decay the LR

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.2f} seconds")

    # 8) Plot training loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), train_losses, marker="o", label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MAE)")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.show()

    # 9) Visualize one validation patch (noisy vs. denoised vs. clean) on a single band
    example_noisy, example_clean = next(iter(val_loader))
    example_noisy = example_noisy.to(device)
    example_clean = example_clean.to(device)

    model.eval()
    with torch.no_grad():
        example_output = model(example_noisy)

    # Convert the first sample (index 0) to NumPy for plotting
    input_patch = example_noisy[0].cpu().numpy()   # shape (200, 64, 64)
    clean_patch = example_clean[0].cpu().numpy()
    output_patch = example_output[0].cpu().numpy()

    # Pick a band to visualize, e.g. band_idx = 0
    band_idx = 0
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(input_patch[band_idx], cmap="gray")
    plt.title("Noisy Input Patch (Band 0)")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(output_patch[band_idx], cmap="gray")
    plt.title("Reconstructed Patch (Band 0)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(clean_patch[band_idx], cmap="gray")
    plt.title("Clean Patch (Band 0)")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    main()