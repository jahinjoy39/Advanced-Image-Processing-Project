import os
from pathlib import Path
import argparse

import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
import torchio as tio
from omegaconf import OmegaConf

from diffusers import UNet2DModel, DDPMScheduler

#######################################################################
# ----------------------------- CONFIG ------------------------------- #
#######################################################################

def load_config(yaml_path):
    cfg = OmegaConf.load(yaml_path)
    return cfg

#######################################################################
# ------------- SAME PREPROCESSING AS NiftiWrapperDataset ----------- #
#######################################################################

def load_and_preprocess_nifti(nifti_path, image_size, center):
    """
    Returns a tensor of shape (1, H, W)
    using the SAME logic as your NiftiWrapperDataset.
    """
    img = nib.load(str(nifti_path))
    data = img.get_fdata()

    # torchio subject
    subject = tio.Subject(image=tio.ScalarImage(tensor=data[None, ...]))
    transform = tio.Compose([
        tio.Resize((image_size, image_size, image_size)),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])
    subject = transform(subject)
    vol = subject["image"].data   # shape: (1, H, W, D)

    center_slice_idx = vol.shape[3] // 2
    slice_2d = vol[:, :, :, center_slice_idx]  # (1, H, W)

    if center:
        slice_2d = slice_2d * 2 - 1

    return slice_2d.float(), img.affine


#######################################################################
# ------------------ SAMPLING LOOP (from train2.py) ------------------ #
#######################################################################

def run_sampling(unet, scheduler, input_img, timesteps, guidance_scale, center):
    """
    Fully reproduces your anom_inference() sampling loop,
    but minimal and standalone.
    """
    device = next(unet.parameters()).device
    input_img = input_img.to(device)

    # Start from pure noise
    noise = torch.randn_like(input_img)

    for t in timesteps:
        if guidance_scale > 0:
            noise_input = torch.cat([noise]*2)
            conditional_input = torch.cat([
                torch.randn_like(input_img),  # unconditional
                input_img                    # conditional
            ])
        else:
            noise_input = noise
            conditional_input = input_img

        noise_model_input = scheduler.scale_model_input(noise_input, t)
        model_input = torch.cat([noise_model_input, conditional_input], dim=1)

        with torch.no_grad():
            noise_pred = unet(model_input, t).sample

        # Guidance
        if guidance_scale > 0:
            eps_uncond, eps_cond = noise_pred.chunk(2)
            noise_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # DDPM update
        step_output = scheduler.step(noise_pred, t, noise)
        noise = step_output.prev_sample

    # Final restored
    restored = noise
    if center:
        restored = restored / 2 + 0.5
    restored = restored.clamp(0, 1)
    return restored


#######################################################################
# ------------------------ MAIN EVAL SCRIPT -------------------------- #
#######################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to omega_config_big_split.yaml")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint folder containing 'unet' or 'unet_ema'")
    parser.add_argument("--nifti_dir", required=True,
                        help="Directory containing input NIfTI volumes")
    parser.add_argument("--out_dir", required=True,
                        help="Where to save restored NIfTIs")
    parser.add_argument("--use_ema", action="store_true",
                        help="Load EMA weights if available")
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--img_channels", type=int, required=True)
    parser.add_argument("--center", action="store_true", default=False)
    args = parser.parse_args()
    upd_config = args

    # Load config
    h_config = load_config(args.config)
    
    if not h_config.unet.get("sample_size"):
        h_config.unet.sample_size = upd_config.image_size

# 2. in_channels = 2 * img_channels
    if not h_config.unet.get("in_channels"):
        h_config.unet.in_channels = 2 *upd_config.img_channels

# 3. out_channels = img_channels
    if not h_config.unet.get("out_channels"):
        h_config.unet.out_channels = upd_config.img_channels
    
    image_size = h_config.unet.sample_size
    in_channels = h_config.unet.in_channels
    out_channels = h_config.unet.out_channels
    validation_timesteps = h_config.validation_timesteps
    guidance = h_config.validation_guidance
    center = upd_config.center

    # Load UNet
    print("Loading UNet...")
    unet = UNet2DModel(**OmegaConf.to_container(h_config.unet, resolve=True))
    checkpoint_dir = Path(args.checkpoint)

    if args.use_ema:
        weight_dir = checkpoint_dir / "unet_ema"
    else:
        weight_dir = checkpoint_dir / "unet"

    if not weight_dir.exists():
        raise FileNotFoundError(f"Missing UNet weights at: {weight_dir}")

    unet = UNet2DModel.from_pretrained(checkpoint_dir, subfolder=weight_dir.name)
    unet = unet.eval().cuda()

    # Load DDPM Scheduler
    scheduler = DDPMScheduler()
    if h_config.prediction_type is not None:
        scheduler.register_to_config(prediction_type=h_config.prediction_type)

    scheduler.set_timesteps(validation_timesteps)
    timesteps = scheduler.timesteps

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over NIfTI files
    nifti_paths = sorted(Path(args.nifti_dir).glob("*.nii*"))
    print(f"Found {len(nifti_paths)} NIfTI files.")

    for nii_path in nifti_paths:
        print(f"\nProcessing {nii_path.name}")

        # Load + preprocess
        slice_2d, affine = load_and_preprocess_nifti(
            nii_path, image_size=image_size, center=center
        )

        # Add batch dimension
        slice_in = slice_2d.unsqueeze(0).cuda()  # (1, 1, H, W)

        # Run sampling
        restored = run_sampling(
            unet,
            scheduler,
            slice_in,
            timesteps,
            guidance_scale=guidance,
            center=center
        )  # (1, 1, H, W)

        restored_np = restored.squeeze().cpu().numpy()

        # Save as NIfTI (1-slice volume)
        restored_vol = np.zeros((1, restored_np.shape[0], restored_np.shape[1]))
        restored_vol[0] = restored_np

        out_path = out_dir / f"{nii_path.stem}_restored.nii.gz"
        nib.save(nib.Nifti1Image(restored_vol, affine), out_path)

        print(f"Saved restored NIfTI to {out_path}")


if __name__ == "__main__":
    main()

