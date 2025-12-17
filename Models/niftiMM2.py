import os
import numpy as np
import nibabel as nib
from glob import glob
import torch
import torch.nn.functional as F

from UPD_study.utilities.common_data import BaseDataset


def extract_subject_id(path):
    fname = os.path.basename(path)
    parts = fname.split("-")
    return "-".join(parts[:4])


class NiftiWrapperDataset(BaseDataset):
    """
    Multimodal T1/T2 dataset.
    Returns (img, target) where both = slice (2, H, W)
    The corruption pipeline modifies img; target stays clean.
    """

    def __init__(self, upd_config, train=True):
        super().__init__(upd_config)

        self.train = train
        self.target_size = upd_config.image_size   # 128
        self.center = False

        self.t1_dir = upd_config.t1_path
        self.t2_dir = upd_config.t2_path

        # Load file lists
        t1_files = sorted(glob(os.path.join(self.t1_dir, "*.nii*")))
        t2_files = sorted(glob(os.path.join(self.t2_dir, "*.nii*")))

        # match subjects
        t2_dict = {extract_subject_id(p): p for p in t2_files}

        self.paired = []
        for t1 in t1_files:
            sid = extract_subject_id(t1)
            if sid in t2_dict:
                self.paired.append((t1, t2_dict[sid]))

        if len(self.paired) == 0:
            raise ValueError("No matched T1/T2 subjects found.")

        print(f"[niftiMM] Matched {len(self.paired)} subjects.")

    def __len__(self):
        return len(self.paired)

    # required
    def get_sample(self, idx):
        t1_path, t2_path = self.paired[idx]

        # load volumes
        t1 = nib.load(t1_path).get_fdata().astype(np.float32)
        t2 = nib.load(t2_path).get_fdata().astype(np.float32)

        # normalize
        def norm(v):
            m, s = v.mean(), v.std()
            return (v - m) / (s + 1e-6)

        t1 = norm(t1)
        t2 = norm(t2)

        # stack → (2, H, W, D)
        vol = np.stack([t1, t2], axis=0)
        D = vol.shape[-1]

        # slice index
        if self.train:
            sd = np.random.randint(0, D)
        else:
            sd = D // 2

        slice2d = vol[:, :, :, sd]  # (2,H,W)

        # convert to torch so we can resize
        slice2d = torch.from_numpy(slice2d).float().unsqueeze(0)   # (1,2,H,W)

        # resize → (1,2,128,128)
        slice2d = F.interpolate(slice2d, size=(self.target_size, self.target_size),
                                mode="bilinear", align_corners=False)

        slice2d = slice2d.squeeze(0)  # (2,128,128)

        # return (input, target)
        # the augmentation pipeline will corrupt input; target remains the clean version
        img = slice2d.numpy().astype(np.float32)
        target = img.copy()

        return img, target


def get_datasets_nifti(upd_config, train=True):
    train_dataset = NiftiWrapperDataset(upd_config, train=True)
    val_dataset = NiftiWrapperDataset(upd_config, train=False)
    return train_dataset, val_dataset

