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

def load_and_preprocess_nifti(t1_nifti_path, t2_nifti_path, image_size, center):
    """
    Returns a stacked tensor of shape (2, H, W)
    using the SAME logic as your NiftiWrapperDataset.
    """
    # 1. Load data
    t1_img = nib.load(str(t1_nifti_path))
    t2_img = nib.load(str(t2_nifti_path))

    t1_data = t1_img.get_fdata().astype(np.float32)
    t2_data = t2_img.get_fdata().astype(np.float32)
    original_depth = t1_data.shape[0]
    # 2. Define Normalization function (Z-score)
    def norm(vol):
        m = vol.mean()
        s = vol.std()
        return (vol - m) / (s + 1e-6)

    t1_data = norm(t1_data)
    t2_data = norm(t2_data)

    # 3. Create torchio subject from stacked data (D, H, W, C) for uniform transformation
    # We stack T1 and T2 to ensure the same transformation is applied to both volumes
    stacked_vol = np.stack([t1_data, t2_data], axis=-1) # (D, H, W, 2)

    vol_c_first = stacked_vol.transpose((3, 0, 1, 2))

    subject = tio.Subject(image=tio.ScalarImage(tensor=vol_c_first)) # (1, D, H, W, 2)

    transform = tio.Compose([
        # Ensure TIO handles the 3D data plus the channel dimension correctly
        tio.Resize((original_depth, image_size, image_size)), # Resize spatial H and W
        # ZNormalization is applied earlier via the manual function
    ])
    subject = transform(subject)
    vol = subject["image"].data # shape: (1, D, H, W, 2) or similar depending on TIO version

    # 4. Final Slicing: Pick center slice (using the depth axis, index 1)
    center_slice_idx = vol.shape[1] // 2
    slice_2d = vol [:, :, :, center_slice_idx]  # shape: (H, W, 2)

    # Transpose to Channel-first format: (2, H, W)
    #slice_2d = slice_2d.permute(2, 0, 1)

    if center:
        slice_2d = slice_2d * 2 - 1

    # Return the 2-channel slice and the affine matrix from the T1 volume
    return slice_2d.float(), t1_img.affine

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
    parser.add_argument("--t1_dir", required=True,
                        help="Directory containing T1 NIfTI files")
    parser.add_argument("--t2_dir", required=True,
                        help="Directory containing T2 NIfTI files")
    #parser.add_argument("--nifti_dir", required=True,
    #                    help="Directory containing input NIfTI volumes")
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
    t1_paths = sorted(Path(args.t1_dir).glob("*.nii*"))
    t2_paths = sorted(Path(args.t2_dir).glob("*.nii*"))

    def extract_subject_id(path):
        fname = os.path.basename(path)
        parts = fname.split("-")
        return "-".join(parts[:4])

    t2_dict = {extract_subject_id(p): p for p in t2_paths}
    paired_paths = []

    for t1_path in t1_paths:
        sid = extract_subject_id(t1_path)
        if sid in t2_dict:
            paired_paths.append((t1_path, t2_dict[sid]))

    print(f"Found {len(paired_paths)} paired subjects.")

    for t1_path, t2_path in paired_paths:
        subject_id = extract_subject_id(str(t1_path))
        print(f"\nProcessing {subject_id}")

        # Load + preprocess
        slice_2d, affine = load_and_preprocess_nifti(
            t1_path, t2_path, image_size=image_size, center=center
        )

        # Add batch dimension
        # slice_in is the CORRUPTED/CONDITIONING INPUT to the UNet
        slice_in = slice_2d.unsqueeze(0).cuda()  # (1, C, H, W)

        # Run sampling
        restored = run_sampling(
            unet,
            scheduler,
            slice_in,
            timesteps,
            guidance_scale=guidance,
            center=center
        )  # (1, C, H, W)

        # --- CONVERSION AND SAVING ---

        # 1. Convert Restored Output (Tensor -> NumPy)
        # restored_np shape is (C, H, W)
        restored_np = restored.squeeze(0).cpu().numpy()

        # 2. Convert Input Slice (Tensor -> NumPy)
        # input_np shape is (C, H, W). We convert it back from the slice_in tensor.
        input_np = slice_in.squeeze(0).cpu().numpy()

        # Save each modality separately
        for chan in range(input_np.shape[0]):
            modality_data_input = input_np[chan, :, :]
            modality_data_restored = restored_np[chan, :, :]

            modality_name = "T1" if chan == 0 else "T2"

            # Create the 1-slice volume (1, H, W)
            input_vol = modality_data_input[np.newaxis, :, :]
            restored_vol = modality_data_restored[np.newaxis, :, :]

            # Create output paths
            base_name = f"{subject_id}_{modality_name}"

            # --- SAVE THE INPUT SLICE (CORRUPTED) ---
            out_path_input = out_dir / f"{base_name}_input.nii.gz"
            nib.save(nib.Nifti1Image(input_vol, affine), out_path_input)

            # --- SAVE THE RESTORED SLICE ---
            out_path_restored = out_dir / f"{base_name}_restored.nii.gz"
            nib.save(nib.Nifti1Image(restored_vol, affine), out_path_restored)

            print(f"Saved {modality_name} input to {out_path_input}")
            print(f"Saved {modality_name} restoration to {out_path_restored}")
if __name__ == "__main__":
    main()

