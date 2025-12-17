# nifti_data_loader.py

import logging
from pathlib import Path

import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchio as tio
from UPD_study.utilities.common_data import BaseDataset # Import the base class

logger = logging.getLogger(__name__)

class NiftiWrapperDataset(BaseDataset):
    """
    Wraps the NIfTI loading and extraction logic, inheriting from BaseDataset
    to ensure compatibility with the original training script.
    """
    def __init__(self, upd_config, train: bool = True):
        # The parent class (BaseDataset) is likely used for other metadata/configs.
        super().__init__(upd_config)
        #self.center = upd_config.center # [-1, 1] or [0, 1]
        
        t1_dir = Path(upd_config.t1_path)
        t2_dir = Path(upd_config.t2_path)

        t1_files = sorted(list(t1_dir.glob("*.nii*")))
        t2_files = sorted(list(t2_dir.glob("*.nii*")))

        def extract_id(path):
            # Example: BraTS-SSA-00002-000-t1c.nii.gz → BraTS-SSA-00002-000
            name = path.name
            parts = name.split("-")
            return "-".join(parts[:4])

        t1_dict = {extract_id(f): f for f in t1_files}
        t2_dict = {extract_id(f): f for f in t2_files}

        self.ids = sorted(set(t1_dict.keys()) & set(t2_dict.keys()))
        self.t1_dict = t1_dict
        self.t2_dict = t2_dict

        if len(self.ids) == 0:
            raise ValueError("No matched T1/T2 subjects found")

        logger.info(f"Found {len(self.ids)} matched T1/T2 subjects.")

        self.image_size = upd_config.image_size
        self.center = upd_config.center

        self.tio_transform = tio.Resize((self.image_size,
                                         self.image_size,
                                         self.image_size))


        # IMPORTANT: The original script expects the dataset to have a .aug_fn attribute
        # which is set *after* the dataset is instantiated in the main script.
        self.aug_fn = None 

    def norm(x):
        """Simple Z-score normalization."""
        x = x.astype(np.float32)
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-8)
    
    def load_and_stack(self, subj_id):
        t1 = nib.load(self.t1_dict[subj_id]).get_fdata().astype(np.float32)
        t2 = nib.load(self.t2_dict[subj_id]).get_fdata().astype(np.float32)

        # Normalize independently
        t1 = norm(t1)
        t2 = norm(t2)

        # Stack modalities → shape (2, H, W, D)
        vol = np.stack([t1, t2], axis=0)

        return vol


    def __len__(self):
        # We assume each *file* contributes one sample (the central slice) for training.
        return len(self.ids)

    def __getitem__(self, idx):
        subj_id = self.ids[idx]

        vol = self.load_and_stack(subj_id)

        # Convert to TorchIO subject with MULTICHANNEL image, NOT ScalarImage
        subject = tio.Subject(
            image=tio.Image(tensor=vol, type=tio.INTENSITY)
        )

        # Apply resize transform
        subject = self.tio_transform(subject)
        vol = subject["image"].data  # (2, H, W, D)

        # Extract center slice
        d = vol.shape[-1]
        mid = d // 2
        slice_2d = vol[:, :, :, mid]   # shape → (2, H, W)

        # Scale to [-1,1] if center=True
        if self.center:
            slice_2d = slice_2d * 2 - 1

        slice_2d = torch.from_numpy(slice_2d).float()

        if self.aug_fn is None:
            return slice_2d, slice_2d

        img_np = slice_2d.cpu().numpy()

        if self.center:
            img_for_aug = (img_np / 2) + 0.5
        else:
            img_for_aug = img_np

        corrupted, target = self.aug_fn(img_for_aug, self)
        if self.center:
            corrupted = corrupted * 2 - 1
            target = target * 2 - 1

        return (
            torch.from_numpy(corrupted).float(),
            torch.from_numpy(target).float()
        )

def get_datasets_nifti(upd_config, train: bool = True):
    """
    Mimics the original get_datasets_* functions, returning train and val instances.
    """
    # For simplicity, we use the same dataset class for train and val/test, 
    # but the calling script will handle the different aug_fn instances.
    
    # NOTE: The original script loads one *main* train/val pair, and several 
    # *external blending* datasets. We only replace the *main* dataset here.
    train_dataset = NiftiWrapperDataset(upd_config, train=True)
    val_dataset = NiftiWrapperDataset(upd_config, train=False)
    
    return train_dataset, val_dataset
