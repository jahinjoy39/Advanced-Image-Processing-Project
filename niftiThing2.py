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
        self.center = upd_config.center # [-1, 1] or [0, 1]
        
        # Determine the data path
        data_dir = Path(upd_config.nifti_data_path)

        self.data_files = list(data_dir.glob("*.nii*"))
        if not self.data_files:
            logger.error(f"No NIfTI files found in {data_dir}")
            raise FileNotFoundError(f"No NIfTI files found in {data_dir}")
        
        self.image_size = upd_config.image_size
        logger.info(f"NIfTI dataset initialized with {len(self.data_files)} files.")

        # Transformations for consistent processing
        self.tio_transform = tio.Compose([
            # Resize the spatial dimensions (H, W) but keep all slices (D)
            tio.Resize((self.image_size, self.image_size, self.image_size)), 
            # Simple ZNormalization 
            tio.ZNormalization(masking_method=tio.ZNormalization.mean), 
        ])

        # IMPORTANT: The original script expects the dataset to have a .aug_fn attribute
        # which is set *after* the dataset is instantiated in the main script.
        self.aug_fn = None 
    
    def get_nifti_affine(self, idx: int):
        """Retrieves the original NIfTI affine matrix for the file at idx."""
        file_path = self.data_files[idx]
        img_nib = nib.load(file_path)
        # We need the 4x4 affine matrix
        return img_nib.affine

    def get_sample(self, idx: int):
        """
        This method is required by the BaseDataset Abstract Base Class.
        It encapsulates the logic to load and process a single NIfTI slice.
        """
        file_path = self.data_files[idx]
        img_nib = nib.load(file_path)
        img_data = img_nib.get_fdata()

        # 1. Convert to torchio Subject and apply transformation
        subject = tio.Subject(image=tio.ScalarImage(tensor=img_data[None, ...])) 
        transformed_subject = self.tio_transform(subject)
        image_tensor = transformed_subject['image'].data

        # 2. Extract a 2D Slice
        depth = image_tensor.shape[1]
        center_slice_idx = depth // 2
        image_2d = image_tensor[:, center_slice_idx, ...]

        # 3. Apply Centering/Normalization
        if self.center:
            image_2d = image_2d * 2 - 1
        
        return image_2d.float() # Return the clean image tensor

    def __len__(self):
        # We assume each *file* contributes one sample (the central slice) for training.
        return len(self.data_files)

    def __getitem__(self, idx):
        # 1. Get the clean image sample using the required abstract method
        clean_image = self.get_sample(idx)
        image_for_aug = clean_image.cpu().numpy()
        # 2. Apply the RestorationTask (Corruption/Target Generation)
        # self.aug_fn is the RestorationTask instance set in the main script.
        if self.aug_fn is not None:
            # Pass the NumPy array and the dataset object (self)
            # The output will be two NumPy arrays (corrupted_img, target_img).
            corrupted_img, target_img = self.aug_fn(image_for_aug, self)

            # 3. Convert outputs back to PyTorch Tensors for the DataLoader
            return torch.from_numpy(corrupted_img).float(), torch.from_numpy(target_img).float()

        else:
            # Fallback must return PyTorch Tensors
            return clean_image, clean_image

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
