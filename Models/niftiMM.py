import os
import numpy as np
import nibabel as nib
import torch
from glob import glob
import torch.nn.functional as F
from UPD_study.utilities.common_data import BaseDataset


def extract_subject_id(path):
    """
    Extract shared subject prefix from filenames like:
    BraTS-SSA-00002-000-t1c.nii.gz → BraTS-SSA-00002-000
    """
    fname = os.path.basename(path)
    parts = fname.split("-")
    return "-".join(parts[:4])


class NiftiWrapperDataset(BaseDataset):
    """
    Multimodal T1/T2 dataset.
    Produces slices shape: (2, H, W).
    """

    def __init__(self, upd_config, train=True):
        # 🔥 FIX: pass config to BaseDataset
        super().__init__(upd_config)

        self.train = train
        self.center = False  # we apply our own normalization manually

        self.t1_dir = upd_config.t1_path
        self.t2_dir = upd_config.t2_path

        # ---------------------------------
        # Load file lists
        # ---------------------------------
        t1_files = sorted(glob(os.path.join(self.t1_dir, "*.nii*")))
        t2_files = sorted(glob(os.path.join(self.t2_dir, "*.nii*")))

        t2_dict = {extract_subject_id(p): p for p in t2_files}

        self.paired = []
        for t1_path in t1_files:
            sid = extract_subject_id(t1_path)
            if sid in t2_dict:
                self.paired.append((t1_path, t2_dict[sid]))

        if len(self.paired) == 0:
            raise ValueError(
                "❗ No matched T1/T2 subjects found. Check filenames."
            )

        print(f"[niftiMM] Successfully matched {len(self.paired)} subjects.")

    def __len__(self):
        return len(self.paired)

    # ----------------------------------------------------
    # REQUIRED BY BaseDataset — must return (input, target)
    # ----------------------------------------------------
    def get_sample(self, idx):
        t1_path, t2_path = self.paired[idx]

        # Load
        t1 = nib.load(t1_path).get_fdata().astype(np.float32)
        t2 = nib.load(t2_path).get_fdata().astype(np.float32)

        # Normalize independently
        def norm(vol):
            m = vol.mean()
            s = vol.std()
            return (vol - m) / (s + 1e-6)

        t1 = norm(t1)
        t2 = norm(t2)

        # Stack → (2, H, W, D)
        stacked = np.stack([t1, t2], axis=0)
        D = stacked.shape[-1]

        # Select slice
        if self.train:
            slice_idx = np.random.randint(0, D)
        else:
            slice_idx = D // 2

        slice_2d = stacked[:, :, :, slice_idx]  # shape (2, H, W)

        slice_torch = torch.from_numpy(slice_2d).float()
        target_size = 84
        if slice_torch.shape[1] != target_size:
            # Add batch dimension (1, C, H, W) for interpolation
            slice_torch = slice_torch.unsqueeze(0)

            slice_torch = F.interpolate(
                slice_torch,
                size=(target_size, target_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0) # Remove batch dimension (C, H, W)
        print(f"Slice Shape: {slice_torch.shape}")
        print(f"Slice Min: {slice_torch.min():.2f}, Slice Max: {slice_torch.max():.2f}")
        #slice_2d = torch.from_numpy(slice_2d.copy()).float()
        #slice_torch = torch.from_numpy(slice_2d).float()
        return slice_torch.cpu().numpy().astype(np.float32)  # (corrupted, target)
    
    def __getitem__(self, idx):
        # 1. Get the clean image sample (returns NumPy array)
        clean_image_np = self.get_sample(idx)
        
        # 2. Get the affine matrix (NumPy array)
        #affine_matrix = self.get_nifti_affine(idx)

        # The subsequent logic relies on the parent's __getitem__ 
        # (common_data.py) calling self.aug_fn. 
        
        # We must manually call the augmentation here to ensure we control the 
        # data flow and the required three items (corrupted, target, affine) are returned.
        
        # Check if the augmentation task is set up
        if self.aug_fn is not None:
            print("In augmentation loop")  
            # --- Perform Augmentation (Requires NumPy) ---
            image_for_aug = clean_image_np 
            
            # 1. Augmentation Call
            # This returns two NumPy arrays (corrupted_img, target_img)
            corrupted_img, target_img = self.aug_fn(image_for_aug, self)
            
            # 2. Final Centering/Scaling (If necessary for diffusion model input)
            # Assuming the task outputs are [0, 1] normalized, and diffusion model needs [-1, 1].
            if self.center: 
                corrupted_img = corrupted_img * 2 - 1 
                target_img = target_img * 2 - 1
            
            # 3. Convert final outputs back to PyTorch Tensors
            corrupted_tensor = torch.from_numpy(corrupted_img).float()
            target_tensor = torch.from_numpy(target_img).float()
            
            # Return (corrupted, target, affine)
            return corrupted_tensor, target_tensor 
        
        else:
            # Fallback if augmentation is not set (e.g., during some initialization)
            clean_tensor = torch.from_numpy(clean_image_np).float()
            return clean_tensor, clean_tensor, affine_matrix

# ----------------------------------------------------
# Factory function for trainMM.py
# ----------------------------------------------------
def get_datasets_nifti(upd_config, train=True):
    train_dataset = NiftiWrapperDataset(upd_config, train=True)
    val_dataset = NiftiWrapperDataset(upd_config, train=False)
    return train_dataset, val_dataset

