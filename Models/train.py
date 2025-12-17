# Refactored for simplicity and NIfTI data loading
# Removed dependencies: accelerate, wandb, UPD_study.* imports, and evaluation logic.

import logging
import math
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# Imports for NIfTI processing
import nibabel as nib
import torchio as tio
from einops import rearrange
from omegaconf import OmegaConf, DictConfig

import diffusers
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

# Removed check_min_version("0.22.0.dev0") as it's not strictly necessary for a simple script
logger = logging.getLogger(__name__)

# --- 1. Custom NIfTI Dataset Implementation ---

class NiftiDataset(Dataset):
    """A simple PyTorch Dataset for loading NIfTI files."""
    def __init__(self, data_dir: str, image_size: int = 256, center: bool = True):
        self.data_files = list(Path(data_dir).glob("*.nii*"))
        self.image_size = image_size
        self.center = center
        # Transformations for consistent processing
        self.transform = tio.Compose([
            tio.Resize(image_size),
            tio.ZNormalization(masking_method=tio.ZNormalization.mean), # Simple normalization
            # ToTensor already applied when loading subject
        ])
        logger.info(f"Found {len(self.data_files)} NIfTI files.")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        img_nib = nib.load(file_path)
        img_data = img_nib.get_fdata()
        
        # Convert to torchio Subject and apply transformation
        # Assuming 3D images (D, H, W)
        subject = tio.Subject(image=tio.ScalarImage(tensor=img_data[None, ...])) # Add channel dimension
        transformed_subject = self.transform(subject)
        image_tensor = transformed_subject['image'].data
        
        # Select the central slice for 2D training (common for medical image U-Nets)
        # You may need to adjust this based on your NIfTI data's dimensions (D, H, W)
        depth = image_tensor.shape[1]
        center_slice_idx = depth // 2
        
        # Grab the central slice (C, H, W)
        image_2d = image_tensor[:, center_slice_idx, ...]

        if self.center:
            # Scale to [-1, 1] as used in the original script's context
            image_2d = image_2d * 2 - 1
            
        # The original script uses an augmentation function (RestorationTask) to create 
        # a (corrupted, target) pair. Since we removed that, we'll just use the original
        # image as both corrupted (input) and target for the loss calculation.
        # This trains a denoising autoencoder-like model.
        corrupted = image_2d.float()
        target = image_2d.float()
        
        return corrupted, target

# --- 2. Simplified Argument Parsing and Configuration ---

def parse_args():
    parser = ArgumentParser()
    # Basic configuration for UNet and training
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--img_channels", type=int, default=1, help="Number of image channels (e.g., 1 for grayscale).")
    parser.add_argument("--nifti_data_path", type=str, required=True, help="Path to the directory containing NIfTI files.")
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="diffusion_output")

    # UNet and Diffuser specific settings (minimal subset)
    parser.add_argument("--prediction_type", type=str, default="epsilon", choices=["epsilon", "v_prediction"])
    parser.add_argument("--center_images", type=str, default="True", choices=["True", "False"], 
                        help="If True, normalize images to [-1, 1], otherwise [0, 1].")
    
    # Placeholder for configuration that remains as DictConfig for UNet initialization
    h_config = OmegaConf.create({
        "unet": {
            "sample_size": 128,
            "in_channels": 2, # Corrupted + Noisy_Target
            "out_channels": 1,
            "layers_per_block": 2,
            "block_out_channels": (128, 128, 256, 256),
            "down_block_types": (
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            "up_block_types": (
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        },
        "num_train_epochs": 100,
        "learning_rate": 1e-4,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 500,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-6,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "prediction_type": "epsilon",
        "seed": 42,
        "train_batch_size": 4,
        "dataloader_num_workers": 4,
        "gradient_accumulation_steps": 1,
        "resume_from_checkpoint": None,
        "mixed_precision": "no",
        "use_ema": False, # Simplified: set EMA to False
        "snr_gamma": None, # Simplified: set snr_gamma to None
        "max_train_steps": None,
        "log_steps": 100,
        # ... other config values ...
    })
    
    args = parser.parse_args()
    
    # Update h_config with command-line arguments
    h_config.train_batch_size = args.train_batch_size
    h_config.num_train_epochs = args.num_train_epochs
    h_config.learning_rate = args.learning_rate
    h_config.output_dir = args.output_dir
    h_config.prediction_type = args.prediction_type

    h_config.unet.sample_size = args.image_size
    h_config.unet.in_channels = 2 * args.img_channels
    h_config.unet.out_channels = args.img_channels

    args.center = args.center_images.lower() == 'true'

    return args, h_config


# --- 3. Loss Calculation (Copied and Minimaly Altered) ---

def compute_loss(h_config, noise_scheduler, unet, uncond_p, batch, device):
    """
    Computes the diffusion loss. Mostly copied from the original script.
    Removed accelerate-specific logic like gather/logging.
    """
    corrupted, target = batch
    corrupted = corrupted.to(device)
    target = target.to(device)
    batch_size = target.shape[0]

    # Simplified Unconditional Training (uncond_p)
    # The original script had a complex augmentation setup; here we just zero the input
    # 'corrupted' image with probability 'uncond_p'.
    if uncond_p > 0:
        for i in range(batch_size):
            if random.random() < uncond_p:
                corrupted[i] = corrupted[i] * 0.

    noise = torch.randn_like(target)

    # Sample a random timestep for each image
    timesteps = torch.randint(
        0, int(noise_scheduler.config.num_train_timesteps),
        (batch_size,), device=target.device).long()

    # Forward diffusion process
    noisy_target = noise_scheduler.add_noise(target, noise, timesteps)

    # Determine the target for the UNet prediction
    if noise_scheduler.config.prediction_type == "epsilon":
        loss_target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        loss_target = noise_scheduler.get_velocity(target, noise, timesteps)
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    # Predict the residual (noise or velocity)
    # input is noisy target + corrupted image (concatenated on channels)
    model_input = torch.cat([noisy_target, corrupted], dim=1)

    # takes B x 2C x H x W and outputs B x C x H x W
    model_pred = unet(model_input, timesteps).sample

    # Loss calculation (simplified to basic MSE, removing SNR logic for minimal change)
    loss = F.mse_loss(model_pred.float(), loss_target.float(), reduction="mean")

    return loss

# --- 4. Main Training Function (Minimal Alterations) ---

def main():
    args, h_config = parse_args()
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if h_config.seed is not None:
        torch.manual_seed(h_config.seed)

    # Handle the repository creation
    Path(h_config.output_dir).mkdir(parents=True, exist_ok=True)

    # 3. Load Scheduler and Model
    noise_scheduler = DDPMScheduler(
        prediction_type=h_config.prediction_type,
        # Default timesteps
    )
    unet = UNet2DModel(**OmegaConf.to_container(h_config.unet, resolve=True))
    unet.to(device)
    unet.train()

    # 4. Data Loading
    train_dataset = NiftiDataset(
        data_dir=args.nifti_data_path,
        image_size=args.image_size,
        center=args.center
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=h_config.train_batch_size,
        num_workers=h_config.dataloader_num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 5. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=h_config.learning_rate,
        betas=(h_config.adam_beta1, h_config.adam_beta2),
        weight_decay=h_config.adam_weight_decay,
        eps=h_config.adam_epsilon,
    )
    
    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / h_config.gradient_accumulation_steps)
    if h_config.max_train_steps is None:
        h_config.max_train_steps = h_config.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        h_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=h_config.lr_warmup_steps,
        num_training_steps=h_config.max_train_steps,
    )
    
    # 6. Training Loop
    global_step = 0
    uncond_p = 0.1 # Simplified, fixed to 0.1

    logger.info("***** Running training *****")
    progress_bar = tqdm(range(h_config.max_train_steps), desc="Steps")
    
    for epoch in range(h_config.num_train_epochs):
        unet.train()
        train_loss = 0.0
        
        for batch in train_dataloader:
            # Accumulation logic removed for simplicity, assuming gradient_accumulation_steps=1
            loss = compute_loss(h_config, noise_scheduler, unet, uncond_p, batch, device)
            
            # Backpropagate
            loss.backward()
            
            # Simplified gradient clipping (no distributed check)
            torch.nn.utils.clip_grad_norm_(unet.parameters(), h_config.max_grad_norm)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            global_step += 1
            progress_bar.update(1)
            
            train_loss += loss.item()

            if global_step % h_config.log_steps == 0:
                avg_loss = train_loss / h_config.log_steps
                logger.info(f"Step {global_step} | Avg Loss: {avg_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}")
                progress_bar.set_postfix({"loss": avg_loss})
                train_loss = 0.0
                
            if global_step >= h_config.max_train_steps:
                break
                
        if global_step >= h_config.max_train_steps:
            break
            
        # Optional: Save checkpoint at the end of each epoch
        if epoch % 10 == 0 or epoch == h_config.num_train_epochs - 1:
            save_path = Path(h_config.output_dir) / f"checkpoint_epoch_{epoch}"
            unet.save_pretrained(save_path)
            logger.info(f"Saved model checkpoint to {save_path}")


if __name__ == "__main__":
    import random # Import random here since it's used in compute_loss
    main()
